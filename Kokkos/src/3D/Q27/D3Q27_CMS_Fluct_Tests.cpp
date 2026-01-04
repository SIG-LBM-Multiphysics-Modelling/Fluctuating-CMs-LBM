#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include "VTKWriter.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using RNGPool = Kokkos::Random_XorShift64_Pool<>;
using namespace std;

const bool plot_vtk = false;   // set true if you really want VTK in tests

typedef Kokkos::View<double*>    View1DArray;
typedef Kokkos::View<double**>   View2DArray;
typedef Kokkos::View<double***>  View3DArray;
typedef Kokkos::View<double****> View4DArray;

struct D3Q27{
    static constexpr double cs2 = 1./3.;
    static constexpr double cs4 = 1./9.;
    static constexpr int dim = 3;
    static constexpr int np = 27;
    const int cx[np];
    const int cy[np];
    const int cz[np];
    const int opp[np];
    const double wf[np];
    const double b[np];

    D3Q27():
         b{1., 1./3., 1./3., 1./3., 2./9., 2./9., 2./9., 1./9., 1./9., 1./9., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27., 
           1./27., 4./81, 4./81., 4./81., 2./81., 2./81., 2./81., 4./243., 4./243., 4./243., 8./729.},
        cx{0, 1, -1, 0,  0, 0,  0, 1, -1,  1, -1, 1, -1,  1, -1, 0,  0,  0,  0, 1, -1,  1, -1,  1, -1,  1, -1},
        cy{0, 0,  0, 1, -1, 0,  0, 1,  1, -1, -1, 0,  0,  0,  0, 1, -1,  1, -1, 1,  1, -1, -1,  1,  1, -1, -1},
        cz{0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1,  1, -1, -1, 1,  1, -1, -1, 1,  1,  1,  1, -1, -1, -1, -1},
       opp{0, 2,  1, 4,  3, 6,  5,10,  9,  8,  7,14, 13, 12, 11,18, 17, 16, 15, 26, 25, 24,23, 22, 21, 20, 19},
        wf{8./27., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,
            1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,
            1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216.} {}
};

// ---- Make Params runtime (so we can change kBT, rho0, etc.)
struct Params{
    double rho0 = 1.0;
    double U_ref = 0.01;
    int nx = 50;
    int ny = 50;
    int nz = 50;

    double ni = 0.1;
    double tau = ni*3.0 + 0.5;
    double omega = 1.0/(tau);
    double omega1 = 1.0 - omega;

    double T_ref = 1.0;        // set in finalize()
    double kBT = 1.0/3000.0;

    int nsteps = 0;            // set in finalize()
    int n_out  = 0;            // set in finalize()

    void finalize(){
        tau   = ni*3.0 + 0.5;
        omega = 1.0/tau;
        omega1 = 1.0 - omega;
        T_ref = double(ny)/U_ref;
        nsteps = (int)(100.*T_ref);
        n_out  = (int)(1.*T_ref);
        if(n_out < 1) n_out = 1;
    }
};

// ---------- small helper: safe sqrt for device
KOKKOS_INLINE_FUNCTION
double safe_sqrt(double x){
    return sqrt(x > 0.0 ? x : 0.0);
}

// ---------- Stats reduction
struct StatsSimple {
    double mu=0, mv=0, mw=0;
    double u2=0, v2=0, w2=0;
    double maxu=0, maxv=0, maxw=0;
};

static StatsSimple compute_stats_simple(View3DArray u, View3DArray v, View3DArray w, const Params& p)
{
    const double N = double(p.nx) * double(p.ny) * double(p.nz);

    double sumu=0.0, sumv=0.0, sumw=0.0;
    double sumu2=0.0, sumv2=0.0, sumw2=0.0;
    double maxu=0.0, maxv=0.0, maxw=0.0;

    // sums
    Kokkos::parallel_reduce(
        "sum_u_v_w",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            lsum += u(x,y,z);
        },
        Kokkos::Sum<double>(sumu)
    );
    Kokkos::parallel_reduce(
        "sum_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            lsum += v(x,y,z);
        },
        Kokkos::Sum<double>(sumv)
    );
    Kokkos::parallel_reduce(
        "sum_w",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            lsum += w(x,y,z);
        },
        Kokkos::Sum<double>(sumw)
    );

    // sums of squares
    Kokkos::parallel_reduce(
        "sum_u2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            const double U = u(x,y,z);
            lsum += U*U;
        },
        Kokkos::Sum<double>(sumu2)
    );
    Kokkos::parallel_reduce(
        "sum_v2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            const double V = v(x,y,z);
            lsum += V*V;
        },
        Kokkos::Sum<double>(sumv2)
    );
    Kokkos::parallel_reduce(
        "sum_w2",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lsum){
            const double W = w(x,y,z);
            lsum += W*W;
        },
        Kokkos::Sum<double>(sumw2)
    );

    // maxima
    Kokkos::parallel_reduce(
        "max_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lmax){
            lmax = fmax(lmax, fabs(u(x,y,z)));
        },
        Kokkos::Max<double>(maxu)
    );
    Kokkos::parallel_reduce(
        "max_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lmax){
            lmax = fmax(lmax, fabs(v(x,y,z)));
        },
        Kokkos::Max<double>(maxv)
    );
    Kokkos::parallel_reduce(
        "max_w",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z, double& lmax){
            lmax = fmax(lmax, fabs(w(x,y,z)));
        },
        Kokkos::Max<double>(maxw)
    );

    StatsSimple s;
    s.mu = sumu / N;  s.mv = sumv / N;  s.mw = sumw / N;
    s.u2 = sumu2 / N; s.v2 = sumv2 / N; s.w2 = sumw2 / N;
    s.maxu = maxu; s.maxv = maxv; s.maxw = maxw;
    return s;
}

void initial_state(View3DArray rho, View3DArray u, View3DArray v, View3DArray w,
                   View4DArray f1, View4DArray f2, const D3Q27 lattice, const Params p)
{
    Kokkos::parallel_for(
        "InitializeState",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{p.nx,p.ny,p.nz}),
        KOKKOS_LAMBDA(const int x,const int y,const int z) {
            const double R = p.rho0;
            const double U = 0.0, V = 0.0, W = 0.0;

            rho(x,y,z)=R; u(x,y,z)=U; v(x,y,z)=V; w(x,y,z)=W;

            for(int k=0;k<lattice.np;k++){
                const double cu = U*lattice.cx[k] + V*lattice.cy[k] + W*lattice.cz[k];
                const double feq = lattice.wf[k] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(U*U+V*V+W*W));
                f1(x,y,z,k)=feq;
                f2(x,y,z,k)=feq;
            }
        }
    );
    Kokkos::fence();
}

// ------------------------------------------------------------------------------------
// YOUR algoLB (kept), with 2 fixes:
//  1) Params is runtime (use Params p, not constexpr)
//  2) Streaming always periodic using modulo (safer).
// ------------------------------------------------------------------------------------
void algoLB(View4DArray f1, View4DArray f2,
            View3DArray rho, View3DArray u, View3DArray v, View3DArray w,
            const D3Q27 lattice, const Params parameters, RNGPool rng_pool)
{
    Kokkos::parallel_for(
        "LBAlgorithm",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{parameters.nx,parameters.ny,parameters.nz}),
        KOKKOS_LAMBDA(const int x, const int y, const int z) {

            double R=0.0, U=0.0, V=0.0, W=0.0;
            for(int k=0;k<lattice.np;k++){
                const double fk = f1(x,y,z,k);
                R += fk;
                U += fk * lattice.cx[k];
                V += fk * lattice.cy[k];
                W += fk * lattice.cz[k];
            }

            // no forcing here
            U /= R; V /= R; W /= R;

            rho(x,y,z)=R; u(x,y,z)=U; v(x,y,z)=V; w(x,y,z)=W;

            const double U2 = U*U;
            const double V2 = V*V;
            const double W2 = W*W;

            // RNG
            auto rng = rng_pool.get_state();
            const double eta4  = rng.normal();
            const double eta5  = rng.normal();
            const double eta6  = rng.normal();
            const double eta7  = rng.normal();
            const double eta8  = rng.normal();
            const double eta9  = rng.normal();
            const double eta10 = rng.normal();
            const double eta11 = rng.normal();
            const double eta12 = rng.normal();
            const double eta13 = rng.normal();
            const double eta14 = rng.normal();
            const double eta15 = rng.normal();
            const double eta16 = rng.normal();
            const double eta17 = rng.normal();
            const double eta18 = rng.normal();
            const double eta19 = rng.normal();
            const double eta20 = rng.normal();
            const double eta21 = rng.normal();
            const double eta22 = rng.normal();
            const double eta23 = rng.normal();
            const double eta24 = rng.normal();
            const double eta25 = rng.normal();
            const double eta26 = rng.normal();
            rng_pool.free_state(rng);

           // std::vector<double> etas = {0., 0., 0., 0., eta4, eta5, eta6, eta7, eta8, eta9, eta10, eta11, eta12, eta13, eta14,
           //                             eta15, eta16, eta17, eta18, eta19, eta20, eta21, eta22, eta23, eta24, eta25, eta26};
           // std::vector<double> phi = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 
          //                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
            double etas[27] = {0., 0., 0., 0., eta4, eta5, eta6, eta7, eta8, eta9, eta10, eta11, eta12, eta13, eta14,
                              eta15, eta16, eta17, eta18, eta19, eta20, eta21, eta22, eta23, eta24, eta25, eta26};
            double phi[27];
            for(int k=0; k<lattice.np; k++)
            {
                if(k<=3)
                    phi[k] = 0.;
                else if(k>=7 && k<=9)
                    phi[k] = safe_sqrt(R*parameters.kBT*parameters.omega*(2.-parameters.omega)*lattice.b[k]/lattice.cs2);
                else
                   phi[k] = safe_sqrt(R*parameters.kBT*lattice.b[k]/lattice.cs2);
              phi[k] *= etas[k];
            }
            // raw moment combos you used
            double r7 = f1(x,y,z,7) - f1(x,y,z,8) - f1(x,y,z,9) + f1(x,y,z,10) + f1(x,y,z,19)
                         - f1(x,y,z,20) - f1(x,y,z,21) + f1(x,y,z,22) + f1(x,y,z,23) - f1(x,y,z,24)
                          - f1(x,y,z,25) + f1(x,y,z,26);

            double r8 = f1(x,y,z,11) - f1(x,y,z,12) - f1(x,y,z,13) + f1(x,y,z,14) + f1(x,y,z,19)
                         - f1(x,y,z,20) + f1(x,y,z,21) - f1(x,y,z,22) - f1(x,y,z,23) + f1(x,y,z,24)
                         - f1(x,y,z,25) + f1(x,y,z,26);

            double r9 = f1(x,y,z,15 )- f1(x,y,z,16) - f1(x,y,z,17) + f1(x,y,z,18) + f1(x,y,z,19) 
                        + f1(x,y,z,20) - f1(x,y,z,21) - f1(x,y,z,22) - f1(x,y,z,23) - f1(x,y,z,24)
                         + f1(x,y,z,25) + f1(x,y,z,26);

            double k7 = r7 - U*V*R;
            double k8 = r8 - U*W*R;
            double k9 = r9 - V*W*R;

            // ---- collide with noise
            // (safe_sqrt avoids NaN if argument is slightly negative numerically)
            /*const double coeff = R*parameters.kBT*lattice.cs2,
                            coeff1 = coeff*parameters.omega*(2.-parameters.omega);

            k4 = parameters.omega1*k4 + eta4 * safe_sqrt(2.*coeff1);
            k5 = parameters.omega1*k5 + eta5 * safe_sqrt(2.*coeff1);
            k6 = parameters.omega1*k6 + eta6 * safe_sqrt(2.*coeff1);
            k7 = parameters.omega1*k7 + eta7 * safe_sqrt(coeff1);
            k8 = parameters.omega1*k8 + eta8 * safe_sqrt(coeff1);

            double k9  = eta9  * safe_sqrt(coeff);
            double k10 = eta10 * safe_sqrt(2.*coeff*lattice.cs2);
            double k11 = eta11 * safe_sqrt(2.*coeff*lattice.cs2);
            double k12 = eta12 * safe_sqrt(2.*coeff*lattice.cs2);
            double k13 = eta13 * safe_sqrt(2.*coeff*lattice.cs2);
            double k14 = eta14 * safe_sqrt(2.*coeff*lattice.cs2);
            double k15 = eta15 * safe_sqrt(2.*coeff*lattice.cs2);
            double k16 = eta16 * safe_sqrt(coeff*lattice.cs2);
            double k17 = eta17 * safe_sqrt(4.*coeff*lattice.cs4);
            double k18 = eta18 * safe_sqrt(4.*coeff*lattice.cs4);
            double k19 = eta19 * safe_sqrt(4.*coeff*lattice.cs4);
            double k20 = eta20 * safe_sqrt(2.*coeff*lattice.cs4);
            double k21 = eta21 * safe_sqrt(2.*coeff*lattice.cs4);
            double k22 = eta22 * safe_sqrt(2.*coeff*lattice.cs4);
            double k23 = eta23 * safe_sqrt(4.*coeff*lattice.cs4*lattice.cs2);
            double k24 = eta24 * safe_sqrt(4.*coeff*lattice.cs4*lattice.cs2);
            double k25 = eta25 * safe_sqrt(4.*coeff*lattice.cs4*lattice.cs2);
            double k26 = eta26 * safe_sqrt(8.*coeff*lattice.cs4*lattice.cs4);*/
            double k4 = phi[4];
            double k5 = phi[5];
            double k6 = phi[6];
            k7 = parameters.omega1*k7 + phi[7];
            k8 = parameters.omega1*k8 + phi[8];
            k9 = parameters.omega1*k9 + phi[9];

            double k10 = phi[10];
            double k11 = phi[11];
            double k12 = phi[12];
            double k13 = phi[13];
            double k14 = phi[14];
            double k15 = phi[15];
            double k16 = phi[16];
            double k17 = phi[17];
            double k18 = phi[18];
            double k19 = phi[19];
            double k20 = phi[20];
            double k21 = phi[21];
            double k22 = phi[22];
            double k23 = phi[23];
            double k24 = phi[24];
            double k25 = phi[25];
            double k26 = phi[26];

            // ---- reconstruct your r0..r26 (kept exactly)
            double r0 = R;
            double r1 = U*R;
            double r2 = V*R;
            double r3 = W*R;
            double r4 = R*U2 + k4;
            double r5 = R*V2 + k5;
            double r6 = R*W2 + k6;
            r7 = U*V*R + k7;
            r8 = U*W*R + k8;
            r9 = V*W*R + k9;
            double r10 = R*V*U2 + 2.*k7*U + k10 + V*k4;
            double r11 = R*W*U2 + 2.*k8*U + k11 + W*k4;
            double r12 = R*U*V2 + 2.*k7*V + k12 + U*k5;
            double r13 = R*U*W2 + 2.*k8*W + k13 + U*k6;
            double r14 = R*V*W2 + 2.*k9*W + k14 + V*k6;
            double r15 = R*W*V2 + 2.*k9*V + k15 + W*k5;
            double r16 = U*V*W*R + W*k7 + V*k8 + U*k9 + k16;
            double r17 = R*U2*V2 + k5*U2 + 4.*k7*U*V + 2.*k12*U + k4*V2 + 2.*k10*V + k17;
            double r18 = R*U2*W2 + k6*U2 + 4.*k8*U*W + 2.*k13*U + k4*W2 + 2.*k11*W + k18;
            double r19 = R*V2*W2 + k6*V2 + 4.*k9*V*W + 2.*k14*V + k5*W2 + 2.*k15*W + k19;
            double r20 = U*V*W2*R + U*V*k6 + W2*k7 + 2.*V*W*k8 + 2.*U*W*k9 + V*k13 + U*k14 + 2.*W*k16 + k20;
            double r21 = U*V2*W*R + U*W*k5 + 2.*V*W*k7 + V2*k8 + 2.*U*V*k9 + W*k12 + U*k15 + 2.*V*k16 + k21;
            double r22 = U2*V*W*R + V*W*k4 + 2.*U*W*k7 + 2.*U*V*k8 + U2*k9 + W*k10 + V*k11 + 2.*U*k16 + k22;
            double r23 = U2*V*W2*R + V*W2*k4 + U2*V*k6 + 2.*U*W2*k7 + 4.*U*V*W*k8 + 2.*U2*W*k9 + W2*k10 + 2.*V*W*k11 + 2.*U*V*k13 + U2*k14 + 4.*U*W*k16 + V*k18 + 2.*U*k20 + 2.*W*k22 + k23;
            double r24 = U2*V2*W*R + V2*W*k4 + U2*W*k5 + 4.*U*V*W*k7 + 2.*U*V2*k8 + 2.*U2*V*k9 + 2.*V*W*k10 + V2*k11 + 2.*U*W*k12 + U2*k15 + 4.*U*V*k16 + W*k17 + 2.*U*k21 + 2.*V*k22 + k24;
            double r25 = U*V2*W2*R + U*W2*k5 + U*V2*k6 + 2.*V*W2*k7 + 2.*V2*W*k8 + 4.*U*V*W*k9 + W2*k12 + V2*k13 + 2.*U*V*k14 + 2.*U*W*k15 + 4.*V*W*k16 + U*k19 + 2.*V*k20 + 2.*W*k21 + k25;
            double r26 = R*U2*V2*W2 + k6*U2*V2 + 4.*k9*U2*V*W + 2.*k14*U2*V + k5*U2*W2 + 2.*k15*U2*W + k19*U2
                      + 4.*k8*U*V2*W + 2.*k13*U*V2 + 4.*k7*U*V*W2 + 8.*k16*U*V*W + 4.*k20*U*V + 2.*k12*U*W2 + 4.*k21*U*W + 2.*k25*U
                      + k4*V2*W2 + 2.*k11*V2*W + k18*V2 + 2.*k10*V*W2 + 4.*k22*V*W + 2.*k23*V
                      + k17*W2 + 2.*k24*W + k26;

            // populations post-collision (your formulas)
            f1(x, y, z, 0)  = (8*r0)/27. - r26 - (4.*(r4 + r5 + r6))/9. + (2.*(r17 + r18 + r19))/3.;
            f1(x, y, z, 1)  = (2.*r0)/27. - (r5 + r6)/9. + (2.*r1)/9. + (2.*r4)/9. + r19/6. + (r25 + r26)/2. - (r12 + r13 + r17 + r18)/3.;
            f1(x, y, z, 2)  = (2.*r0)/27. - (r5 + r6)/9. - (2.*r1)/9. + (2.*r4)/9. + r19/6. - r25/2. + r26/2. + (r12 + r13)/3. - (r17 + r18)/3.;
            f1(x, y, z, 3)  = (2.*r0)/27. - (r4 + r6)/9. + (2.*r2)/9. + (2.*r5)/9. + r18/6. + (r23 + r26)/2. - (r10 + r14 + r17 + r19)/3.;
            f1(x, y, z, 4)  = (2.*r0)/27. - (r4 + r6)/9. - (2.*r2)/9. + (2.*r5)/9. + r18/6. - r23/2. + r26/2. + (r10 + r14)/3. - (r17 + r19)/3.;
            f1(x, y, z, 5)  = (2.*r0)/27. - (r4 + r5)/9. + (2.*r3)/9. + (2.*r6)/9. + r17/6. + (r24 + r26)/2. - (r11 + r15 + r18 + r19)/3.;
            f1(x, y, z, 6)  = (2.*r0)/27. - (r4 + r5)/9. - (2.*r3)/9. + (2.*r6)/9. + r17/6. - r24/2. + r26/2. + (r11 + r15)/3. - (r18 + r19)/3.;
            f1(x, y, z, 7)  = r0/54. - r6/36. + (r1 + r2 + r4 + r5)/18. + (r7 + r10 + r12 + r17)/6. - (r13 + r14 + r18 + r19)/12. - (r20 + r23 + r25 + r26)/4.;
            f1(x, y, z, 8)  = r0/54. - r1/18. - r6/36. + r13/12. - (r7 + r12)/6. + (r10 + r17)/6. + (r20 + r25)/4. - (r23 + r26)/4. + (r2 + r4 + r5)/18. - (r14 + r18 + r19)/12.;
            f1(x, y, z, 9)  = r0/54. - r2/18. - r6/36. + r14/12. - (r7 + r10)/6. + (r12 + r17)/6. + (r20 + r23)/4. - (r25 + r26)/4. + (r1 + r4 + r5)/18. - (r13 + r18 + r19)/12.;
            f1(x, y, z, 10) = r0/54. - r6/36. - (r1 + r2)/18. + (r4 + r5)/18. - (r10 + r12)/6. + (r7 + r17)/6. + (r13 + r14)/12. - (r18 + r19)/12. - (r20 + r26)/4. + (r23 + r25)/4.;
            f1(x, y, z, 11) = r0/54. - r5/36. + (r1 + r3 + r4 + r6)/18. + (r8 + r11 + r13 + r18)/6. - (r12 + r15 + r17 + r19)/12. - (r21 + r24 + r25 + r26)/4.;
            f1(x, y, z, 12) = r0/54. - r1/18. - r5/36. + r12/12. - (r8 + r13)/6. + (r11 + r18)/6. + (r21 + r25)/4. - (r24 + r26)/4. + (r3 + r4 + r6)/18. - (r15 + r17 + r19)/12.;
            f1(x, y, z, 13) = r0/54. - r3/18. - r5/36. + r15/12. - (r8 + r11)/6. + (r13 + r18)/6. + (r21 + r24)/4. - (r25 + r26)/4. + (r1 + r4 + r6)/18. - (r12 + r17 + r19)/12.;
            f1(x, y, z, 14) = r0/54. - r5/36. - (r1 + r3)/18. + (r4 + r6)/18. - (r11 + r13)/6. + (r8 + r18)/6. + (r12 + r15)/12. - (r17 + r19)/12. - (r21 + r26)/4. + (r24 + r25)/4.;
            f1(x, y, z, 15) = r0/54. - r4/36. + (r2 + r3 + r5 + r6)/18. - (r10 + r11 + r17 + r18)/12. + (r9 + r14 + r15 + r19)/6. - (r22 + r23 + r24 + r26)/4.;
            f1(x, y, z, 16) = r0/54. - r2/18. - r4/36. + r10/12. - (r9 + r14)/6. + (r15 + r19)/6. + (r22 + r23)/4. - (r24 + r26)/4. + (r3 + r5 + r6)/18. - (r11 + r17 + r18)/12.;
            f1(x, y, z, 17) = r0/54. - r3/18. - r4/36. + r11/12. - (r9 + r15)/6. + (r14 + r19)/6. + (r22 + r24)/4. - (r23 + r26)/4. + (r2 + r5 + r6)/18. - (r10 + r17 + r18)/12.;
            f1(x, y, z, 18) = r0/54. - r4/36. - (r2 + r3)/18. + (r5 + r6)/18. + (r10 + r11)/12. + (r9 + r19)/6. - (r14 + r15)/6. - (r17 + r18)/12. + (r23 + r24)/4. - (r22 + r26)/4.;
            f1(x, y, z, 19) = r0/216. + (r1 + r2 + r3 + r4 + r5 + r6)/72. + (r16 + r20 + r21 + r22 + r23 + r24 + r25 + r26)/8. + (r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14 + r15 + r17 + r18 + r19)/24.;
            f1(x, y, z, 20) = r0/216. - r1/72. - (r7 + r8 + r12 + r13)/24. - (r16 + r20 + r21 + r25)/8. + (r22 + r23 + r24 + r26)/8. + (r2 + r3 + r4 + r5 + r6)/72. + (r9 + r10 + r11 + r14 + r15 + r17 + r18 + r19)/24.;
            f1(x, y, z, 21) = r0/216. - r2/72. - (r7 + r9 + r10 + r14)/24. - (r16 + r20 + r22 + r23)/8. + (r21 + r24 + r25 + r26)/8. + (r1 + r3 + r4 + r5 + r6)/72. + (r8 + r11 + r12 + r13 + r15 + r17 + r18 + r19)/24.;
            f1(x, y, z, 22) = r0/216. - (r1 + r2)/72. + (r3 + r4 + r5 + r6)/72. + (r16 + r20 + r24 + r26)/8. - (r21 + r22 + r23 + r25)/8. - (r8 + r9 + r10 + r12 + r13 + r14)/24. + (r7 + r11 + r15 + r17 + r18 + r19)/24.;
            f1(x, y, z, 23) = r0/216. - r3/72. - (r8 + r9 + r11 + r15)/24. - (r16 + r21 + r22 + r24)/8. + (r20 + r23 + r25 + r26)/8. + (r1 + r2 + r4 + r5 + r6)/72. + (r7 + r10 + r12 + r13 + r14 + r17 + r18 + r19)/24.;
            f1(x, y, z, 24) = r0/216. - (r1 + r3)/72. + (r2 + r4 + r5 + r6)/72. + (r16 + r21 + r23 + r26)/8. - (r20 + r22 + r24 + r25)/8. - (r7 + r9 + r11 + r12 + r13 + r15)/24. + (r8 + r10 + r14 + r17 + r18 + r19)/24.;
            f1(x, y, z, 25) = r0/216. - (r2 + r3)/72. + (r1 + r4 + r5 + r6)/72. - (r20 + r21 + r23 + r24)/8. + (r16 + r22 + r25 + r26)/8. - (r7 + r8 + r10 + r11 + r14 + r15)/24. + (r9 + r12 + r13 + r17 + r18 + r19)/24.;
            f1(x, y, z, 26) = r0/216. - (r1 + r2 + r3)/72. + (r4 + r5 + r6)/72. - (r16 + r23 + r24 + r25)/8. + (r20 + r21 + r22 + r26)/8. - (r10 + r11 + r12 + r13 + r14 + r15)/24. + (r7 + r8 + r9 + r17 + r18 + r19)/24.;

            for(int k=0;k<lattice.np;k++){
                const int newx = (x + lattice.cx[k] + parameters.nx) % parameters.nx;
                const int newy = (y + lattice.cy[k] + parameters.ny) % parameters.ny;
                const int newz = (z + lattice.cz[k] + parameters.nz) % parameters.nz;
                f2(newx,newy,newz,k) = f1(x,y,z,k);
            }
        }
    );
    Kokkos::fence();
}

// ----- run one simulation with given params, print stats periodically
static void run_single(const std::string& label, Params p)
{
        namespace fs = std::filesystem;
    fs::path dirname = "./EQUIP";
    bool first_write = true;  // Ensures file is cleared only once


    // Open the output file
    if (fs::exists(dirname)) {
        std::cout << "Directory exists. Overwriting..." << dirname << std::endl;
    }
    else {
        fs::create_directories(dirname);
    }
    fs::path output_path;
    p.finalize();
    const D3Q27 lattice;

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "nx="<<p.nx<<" ny="<<p.ny<<" nz="<<p.nz
              << " rho0="<<p.rho0<<" kBT="<<p.kBT
              << " omega="<<p.omega
              << " T_ref="<<p.T_ref
              << " nsteps="<<p.nsteps
              << "\n";

    View4DArray f1("f1", p.nx,p.ny,p.nz, lattice.np);
    View4DArray f2("f2", p.nx,p.ny,p.nz, lattice.np);
    View3DArray rho("rho", p.nx,p.ny,p.nz);
    View3DArray u("u", p.nx,p.ny,p.nz);
    View3DArray v("v", p.nx,p.ny,p.nz);
    View3DArray w("w", p.nx,p.ny,p.nz);

    RNGPool rng_pool(123456789ULL);

    initial_state(rho,u,v,w,f1,f2,lattice,p);

    const double target = (p.rho0>0.0) ? (p.kBT/p.rho0) : 0.0;

    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
        auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
        auto h_w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), w);
        
        

    for(int it=0; it<=p.nsteps; ++it){
        algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool);
        std::swap(f1,f2);

        if(it % p.n_out == 0){
            output_path = dirname / ("fluid_t_" + std::to_string(it) + ".vtk");
            Kokkos::deep_copy(h_u, u);
            Kokkos::deep_copy(h_v, v);
            Kokkos::deep_copy(h_w, w);
            VTKWriter::write(output_path.string(), p.nx, p.ny, p.nz, h_u, h_v, h_w, p.U_ref);

        auto s = compute_stats_simple(u,v,w,p);
        std::cout << std::setprecision(10)
          << "t/Tref="<<(it/p.T_ref)
          //<< "  <u>="<<s.mu<<" <v>="<<s.mv<<" <w>="<<s.mw
          << "  <u^2>="<<s.u2<<" <v^2>="<<s.v2<<" <w^2>="<<s.w2
          << "  target(kBT/rho)="<<target
          //<< "  max|u|="<<s.maxu<<" max|v|="<<s.maxv<<" max|w|="<<s.maxw
          << "\n";

        }
    }
}

// ---------- CASES
static void case1_zero_noise(){
    Params p;
    p.kBT = 0.0;
    run_single("CASE 1: zero-noise regression (kBT=0)", p);
}

static void case2_equipartition(){
    Params p;
    p.kBT = 1.0/3000.0;
    run_single("CASE 2: equipartition (kBT=1/3000)", p);
}

static void case3_kbt_scaling(){
    std::cout << "\n=== CASE 3: kBT scaling (expect <u^2> ~ kBT/rho) ===\n";
    double kbts[4] = {1.0/6000.0, 1.0/3000.0, 1.0/1500.0, 1.0/750.0};
    for(double kbt : kbts){
        Params p;
        p.kBT = kbt;
        p.finalize();

        const D3Q27 lattice;
        View4DArray f1("f1", p.nx,p.ny,p.nz, lattice.np);
        View4DArray f2("f2", p.nx,p.ny,p.nz, lattice.np);
        View3DArray rho("rho", p.nx,p.ny,p.nz);
        View3DArray u("u", p.nx,p.ny,p.nz);
        View3DArray v("v", p.nx,p.ny,p.nz);
        View3DArray w("w", p.nx,p.ny,p.nz);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p);

        // warmup shorter to keep it fast
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool);
            std::swap(f1,f2);
        }

        auto s = compute_stats_simple(u,v,w,p);
        const double N = double(p.nx)*p.ny*p.nz;
        const double u2 = s.u2, v2 = s.v2, w2 = s.w2;
        const double mean = (u2+v2+w2)/3.0;

        std::cout << std::setprecision(10)
          << "kBT="<<kbt
          << "  <u^2>="<<u2<<" <v^2>="<<v2<<" <w^2>="<<w2
          << "  mean="<<mean
          << "  target="<<(kbt/p.rho0)
          << "\n";
    }
}

static void case4_rho_scaling(){
    std::cout << "\n=== CASE 4: rho scaling (expect <u^2> ~ kBT/rho0) ===\n";
    double rhos[4] = {0.5, 1.0, 2.0, 4.0};
    for(double rho0 : rhos){
        Params p;
        p.rho0 = rho0;
        p.kBT  = 1.0/3000.0;
        p.finalize();

        const D3Q27 lattice;
        View4DArray f1("f1", p.nx,p.ny,p.nz, lattice.np);
        View4DArray f2("f2", p.nx,p.ny,p.nz, lattice.np);
        View3DArray rho("rho", p.nx,p.ny,p.nz);
        View3DArray u("u", p.nx,p.ny,p.nz);
        View3DArray v("v", p.nx,p.ny,p.nz);
        View3DArray w("w", p.nx,p.ny,p.nz);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool);
            std::swap(f1,f2);
        }

        auto s = compute_stats_simple(u,v,w,p);
        const double N = double(p.nx)*p.ny*p.nz;
        const double u2 = s.u2, v2 = s.v2, w2 = s.w2;
        const double mean = (u2+v2+w2)/3.0;

        std::cout << std::setprecision(10)
          << "rho0="<<rho0
          << "  <u^2>="<<u2<<" <v^2>="<<v2<<" <w^2>="<<w2
          << "  mean="<<mean
          << "  target="<<(p.kBT/rho0)
          << "\n";
    }
}

static void case5_tau_sweep(){
    std::cout << "\n=== CASE 5: tau sweep via ni (equipartition at fixed kBT) ===\n";

    // tau values from your table
    double taus[15] = {0.5, 0.5001, 0.5005, 0.501, 0.505, 0.51, 0.55, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0};
    // keep the same kBT used in your other tests
    const double kBT = 1.0/3000.0;

    for(double tau_target : taus){
        Params p;
        p.kBT = kBT;

        // set ni so that tau = 3*ni + 0.5 matches tau_target
        p.ni  = (tau_target - 0.5)/3.0;

        // finalize recomputes tau, omega, etc.
        p.finalize();

        // (optional) sanity check print
        std::cout << std::setprecision(10)
                  << "\n-- target tau=" << tau_target
                  << "  set ni=" << p.ni
                  << "  actual tau=" << p.tau
                  << "  omega=" << p.omega
                  << "\n";

        const D3Q27 lattice;

        View4DArray f1("f1", p.nx,p.ny,p.nz, lattice.np);
        View4DArray f2("f2", p.nx,p.ny,p.nz, lattice.np);
        View3DArray rho("rho", p.nx,p.ny,p.nz);
        View3DArray u("u", p.nx,p.ny,p.nz);
        View3DArray v("v", p.nx,p.ny,p.nz);
        View3DArray w("w", p.nx,p.ny,p.nz);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p);

        // same warmup/run idea as your scaling cases
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool);
            std::swap(f1,f2);
        }

        // report the same stats you already trust
        auto s = compute_stats_simple(u,v,w,p);
        const double mean_u2 = (s.u2 + s.v2 + s.w2)/3.0;

        std::cout << std::setprecision(10)
                  << "tau=" << p.tau
                  << "  <u^2>=" << s.u2
                  << " <v^2>=" << s.v2
                  << " <w^2>=" << s.w2
                  << "  mean=" << mean_u2
                  << "  target(kBT/rho0)=" << (p.kBT/p.rho0)
                  << "\n";
    }
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int which = 0; // 0 -> all
        for(int i=1;i<argc;i++){
            if(strcmp(argv[i],"-case")==0 && i+1<argc){
                which = atoi(argv[i+1]);
            }
        }

        if(which==0){
            case1_zero_noise();
            case2_equipartition();
            case3_kbt_scaling();
            case4_rho_scaling();
            case5_tau_sweep();
        } else if(which==1){
            case1_zero_noise();
        } else if(which==2){
            case2_equipartition();
        } else if(which==3){
            case3_kbt_scaling();
        } else if(which==4){
            case4_rho_scaling();
        } else if(which==5){
            case5_tau_sweep();
        } else {
            std::cout << "Unknown -case. Use 1..4 (or omit to run all).\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
