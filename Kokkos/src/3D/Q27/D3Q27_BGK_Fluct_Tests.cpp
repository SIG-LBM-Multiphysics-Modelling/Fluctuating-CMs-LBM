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
    static constexpr double cs6 = 1./27.;
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
           1./27., 4./81., 4./81., 4./81., 2./81., 2./81., 2./81., 4./243., 4./243., 4./243., 8./729.},
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
    int nx = 20;
    int ny = 10;
    int nz = 15;

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
        nsteps = (int)(10.*T_ref);
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
                   View4DArray f1, View4DArray f2, const D3Q27 lattice, const Params p, View2DArray M)
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
    for(int i=0; i<lattice.np; i++)
    {
      double CX = lattice.cx[i];
      double CY = lattice.cy[i];
      double CZ = lattice.cz[i];
      double CX2 = CX*CX;
      double CY2 = CY*CY;
      double CZ2 = CZ*CZ;
      M(0,i) = 1.;
      M(1,i) = CX;
      M(2,i) = CY;
      M(3,i) = CZ;
      M(4,i) = CX2-lattice.cs2;
      M(5,i) = CY2-lattice.cs2;
      M(6,i) = CZ2-lattice.cs2;
      M(7,i) = CX*CY;
      M(8,i) = CX*CZ;
      M(9,i) = CY*CZ;
      M(10,i) = CX2*CY-lattice.cs2*CY;
      M(11,i) = CX2*CZ-lattice.cs2*CZ;
      M(12,i) = CX*CY2-lattice.cs2*CX;
      M(13,i) = CX*CZ2-lattice.cs2*CX;
      M(14,i) = CY*CZ2-lattice.cs2*CY;
      M(15,i) = CY2*CZ-lattice.cs2*CZ;
      M(16,i) = CX*CY*CZ;
      M(17,i) = CX2*CY2 - lattice.cs2*(CX2+CY2) + lattice.cs4;
      M(18,i) = CX2*CZ2 - lattice.cs2*(CX2+CZ2) + lattice.cs4;
      M(19,i) = CY2*CZ2 - lattice.cs2*(CY2+CZ2) + lattice.cs4;
      M(20,i) = CX*CY*CZ2 - lattice.cs2*CX*CY;
      M(21,i) = CX*CY2*CZ - lattice.cs2*CX*CZ;
      M(22,i) = CX2*CY*CZ - lattice.cs2*CY*CZ;
      M(23,i) = CX2*CY*CZ2 - lattice.cs2*(CX2*CY+CY*CZ2) + lattice.cs4*CY;
      M(24,i) = CX2*CY2*CZ - lattice.cs2*(CX2*CZ+CY2*CZ) + lattice.cs4*CZ;
      M(25,i) = CX*CY2*CZ2 - lattice.cs2*(CX*CY2+CX*CZ2) + lattice.cs4*CX;
      M(26,i) = CX2*CY2*CZ2 - lattice.cs2*(CX2*CY2+CX2*CZ2+CY2*CZ2) + lattice.cs4*(CX2+CY2+CZ2) - lattice.cs6;
    }
    Kokkos::fence();
}

// ------------------------------------------------------------------------------------
void algoLB(View4DArray f1, View4DArray f2,
            View3DArray rho, View3DArray u, View3DArray v, View3DArray w,
            const D3Q27 lattice, const Params parameters, RNGPool rng_pool, View2DArray M)
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

            double coeff = R*parameters.kBT*parameters.omega*(2.-parameters.omega)/lattice.cs2;
            double etas[27] = {0., 0., 0., 0., eta4, eta5, eta6, eta7, eta8, eta9, eta10, eta11, eta12, eta13, eta14,
                              eta15, eta16, eta17, eta18, eta19, eta20, eta21, eta22, eta23, eta24, eta25, eta26};

            for(int l=0;l<lattice.np;l++){
                const double cu = U*lattice.cx[l] + V*lattice.cy[l] + W*lattice.cz[l];
                const double feq = lattice.wf[l] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(U*U+V*V+W*W));
                f1(x,y,z,l) = feq + parameters.omega1*(f1(x,y,z,l)-feq);
                
                double fluctuating_term = 0.;
                for(int k=4; k<lattice.np; k++)
                  fluctuating_term += 1./lattice.b[k]*M(k,l)*etas[k]*sqrt(coeff*lattice.b[k]);

                f1(x,y,z,l) += lattice.wf[l]*fluctuating_term;

                const int newx = (x + lattice.cx[l] + parameters.nx) % parameters.nx;
                const int newy = (y + lattice.cy[l] + parameters.ny) % parameters.ny;
                const int newz = (z + lattice.cz[l] + parameters.nz) % parameters.nz;
                f2(newx,newy,newz,l) = f1(x,y,z,l);
            }
        }
    );
    Kokkos::fence();
}

// ----- run one simulation with given params, print stats periodically
static void run_single(const std::string& label, Params p)
{
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
    View2DArray M("M", 27, 27);

    RNGPool rng_pool(123456789ULL);

    initial_state(rho,u,v,w,f1,f2,lattice,p, M);

    const double target = (p.rho0>0.0) ? (p.kBT/p.rho0) : 0.0;

    for(int it=0; it<=p.nsteps; ++it){
        algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool, M);
        std::swap(f1,f2);

        if(it % p.n_out == 0){
        auto s = compute_stats_simple(u,v,w,p);
        std::cout << std::setprecision(10)
          << "t/Tref="<<(it/p.T_ref)
          << "  <u>="<<s.mu<<" <v>="<<s.mv<<" <w>="<<s.mw
          << "  <u^2>="<<s.u2<<" <v^2>="<<s.v2<<" <w^2>="<<s.w2
          << "  target(kBT/rho)="<<target
          << "  max|u|="<<s.maxu<<" max|v|="<<s.maxv<<" max|w|="<<s.maxw
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
    std::vector<double> kbts = {1.0/6000.0, 1.0/3000.0, 1.0/1500.0, 1.0/750.0};
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
        View2DArray M("M", 27, 27);
  
        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p,M);

        // warmup shorter to keep it fast
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool,M);
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
    std::vector<double> rhos = {0.5, 1.0, 2.0, 4.0};
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
        View2DArray M("M", 27, 27);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p,M);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool, M);
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
        View2DArray M("M", 27, 27);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,w,f1,f2,lattice,p, M);

        // same warmup/run idea as your scaling cases
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,w,lattice,p,rng_pool, M);
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