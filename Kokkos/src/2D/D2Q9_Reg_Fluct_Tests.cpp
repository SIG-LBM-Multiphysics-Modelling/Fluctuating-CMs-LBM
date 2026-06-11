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
#include "VTKWriter2D.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
using namespace std;
using RNGPool = Kokkos::Random_XorShift64_Pool<>;

const bool plot_vtk = true;

typedef Kokkos::View<double*> View1DArray;
typedef Kokkos::View<double**> View2DArray;
typedef Kokkos::View<double***> View3DArray;

struct D2Q9{
  // lattice velocities
    static constexpr double cs = 0.57735026918962576451;
    static constexpr double cs2 = cs*cs;
    static constexpr double cs3 = cs2*cs;
    static constexpr double cs4 = cs3*cs;
    static constexpr double cs6 = cs4*cs2;
    static constexpr double cs8 = cs6*cs2;
    static constexpr int dim = 2;
    static constexpr int np = 9;
    const int cx[np];
    const int cy[np];
    const int opp[np];
    const double wf[np];
    const double b[np];

    D2Q9():
         b{1., 1./3., 1./3., 4./9., 4./9., 1./9., 2./27., 2./27., 4./81.},
        cx{0, 1, 0, -1,  0, 1, -1, -1,  1},
        cy{0, 0, 1,  0, -1, 1,  1, -1, -1},
       opp{0, 3, 4,  1,  2, 7,  8,  5,  6},
        // Weight factors for lattice directions
        wf{4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.} {}
};

struct Params{
            double rho0 = 1.0;
            double U_ref = 0.01;
            int nx = 200;
            int ny = 200;

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
    double mu=0, mv=0;
    double u2=0, v2=0;
    double maxu=0, maxv=0;
};

static StatsSimple compute_stats_simple(View2DArray u, View2DArray v, const Params& p)
{
    const double N = double(p.nx) * double(p.ny);

    double sumu=0.0, sumv=0.0;
    double sumu2=0.0, sumv2=0.0;
    double maxu=0.0, maxv=0.0;

    // sums
    Kokkos::parallel_reduce(
        "sum_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            lsum += u(x,y);
        },
        Kokkos::Sum<double>(sumu)
    );
    Kokkos::parallel_reduce(
        "sum_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            lsum += v(x,y);
        },
        Kokkos::Sum<double>(sumv)
    );

    // sums of squares
    Kokkos::parallel_reduce(
        "sum_u2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            const double U = u(x,y);
            lsum += U*U;
        },
        Kokkos::Sum<double>(sumu2)
    );
    Kokkos::parallel_reduce(
        "sum_v2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            const double V = v(x,y);
            lsum += V*V;
        },
        Kokkos::Sum<double>(sumv2)
    );

    // maxima
    Kokkos::parallel_reduce(
        "max_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax){
            lmax = fmax(lmax, fabs(u(x,y)));
        },
        Kokkos::Max<double>(maxu)
    );
    Kokkos::parallel_reduce(
        "max_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax){
            lmax = fmax(lmax, fabs(v(x,y)));
        },
        Kokkos::Max<double>(maxv)
    );

    StatsSimple s;
    s.mu = sumu / N;  s.mv = sumv / N;
    s.u2 = sumu2 / N; s.v2 = sumv2 / N;
    s.maxu = maxu; s.maxv = maxv;
    return s;
}

void initial_state(View2DArray rho, View2DArray u, View2DArray v, View3DArray f1, View3DArray f2,
                const D2Q9 lattice, const Params parameters, View2DArray M) {

    // Kokkos lambda for initialization
    Kokkos::parallel_for("InitializeState",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {parameters.nx, parameters.ny}),
    KOKKOS_LAMBDA(const int x, const int y) {

        double R = parameters.rho0;
        double U = 0.;
        double V = 0.;
        
        rho(x,y) = R;
        u(x,y) = U;
        v(x,y) = V;
        
        double first_order;

        for (int k = 0; k < lattice.np; k++) {
            first_order = U*lattice.cx[k] + V*lattice.cy[k];
            f1(x,y,k) = f2(x,y,k) = lattice.wf[k] * R  * (1. + 3.*first_order + 4.5*pow(first_order,2) -1.5*(U*U+V*V));
        }
    }
    );
    for(int i=0; i<lattice.np; i++)
    {
      double CX = lattice.cx[i];
      double CY = lattice.cy[i];
      double CX2 = CX*CX;
      double CY2 = CY*CY;
      M(0,i) = 1.;
      M(1,i) = CX;
      M(2,i) = CY;
      M(3,i) = CX2+CY2-2.*lattice.cs2;
      M(4,i) = CX2-CY2;
      M(5,i) = CX*CY;
      M(6,i) = (CX2-lattice.cs2)*CY;
      M(7,i) = (CY2-lattice.cs2)*CX;
      M(8,i) = (CX2-lattice.cs2)*(CY2-lattice.cs2);
    }
    Kokkos::fence();
}

void algoLB(View3DArray f1, View3DArray f2, View2DArray rho, View2DArray u, View2DArray v,
            const D2Q9 lattice, const Params parameters, RNGPool rng_pool, View2DArray M)
{

    // Kokkos parallel loop for the lattice Boltzmann algorithm
    Kokkos::parallel_for("LBAlgorithm",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {parameters.nx, parameters.ny}),
    KOKKOS_LAMBDA(const int x, const int y) {
          
        double R = f1(x,y,0) + f1(x,y,1) + f1(x,y,2) + f1(x,y,3) + f1(x,y,4) + f1(x,y,5) + f1(x,y,6) + f1(x,y,7) + f1(x,y,8);
        double U = (f1(x,y,1) - f1(x,y,3) + f1(x,y,5) - f1(x,y,6) - f1(x,y,7) + f1(x,y,8))/R;
        double V = (f1(x,y,2) - f1(x,y,4) + f1(x,y,5) + f1(x,y,6) - f1(x,y,7) - f1(x,y,8))/R;
        
        double Fx = 0.;
        double Fy = 0.;
        U += 0.5*Fx;
        V += 0.5*Fy;
        rho(x,y) = R;
        u(x,y) = U;
        v(x,y) = V;
        
        double U2 = U*U;
        double V2 = V*V;
        double UV = U*V;

        // ---- RNG: standard normal variables eta_k ~ N(0,1)
        auto rng = rng_pool.get_state();
        const double eta3 = rng.normal();
        const double eta4 = rng.normal();
        const double eta5 = rng.normal();
        const double eta6 = rng.normal();
        const double eta7 = rng.normal();
        const double eta8 = rng.normal();
        rng_pool.free_state(rng);

        double coeff = R*parameters.kBT*parameters.omega*(2.-parameters.omega)/lattice.cs2;
        double etas[9] = {0., 0., 0., eta3, eta4, eta5, eta6, eta7, eta8};
        
        double axx_2 = 0., ayy_2 = 0., axy_2 = 0.;
        for(int k=0; k<lattice.np; k++)
        {
            const double cxk = (double)lattice.cx[k];
            const double cyk = (double)lattice.cy[k];

            // Compact Hermite polynomials for D2 (raw, not normalized by cs)
            // H2:
            const double H2_xx = cxk*cxk - lattice.cs2;
            const double H2_yy = cyk*cyk - lattice.cs2;
            const double H2_xy = cxk*cyk;
            const double cx2 = cxk*cxk, cy2 = cyk*cyk;

            const double first_order = 3.*(U*cxk+V*cyk);
            const double second_order = 4.5*pow(U*cxk+V*cyk,2)-1.5*(U2+V2);
            const double third_order = 1./(2.*lattice.cs6)*(H2_xx *cyk*U2*V+H2_yy*cxk*U*V2);
            const double fourth_order = 1./(4.*lattice.cs8)*(H2_xx*H2_yy*U2*V2);
            const double feq_k = lattice.wf[k]*R*(1.+first_order+second_order+third_order+fourth_order);

            axx_2 += (f1(x, y, k)-feq_k)*H2_xx;
            ayy_2 += (f1(x, y, k)-feq_k)*H2_yy;
            axy_2 += (f1(x, y, k)-feq_k)*H2_xy;

            f1(x, y, k) = feq_k;
        }
          // --- Symmetry
        const double ayx_2 = axy_2;

          // --- RR: build nonequilibrium Hermite coefficients a3 and a4 (isothermal)
          // a3 is fully symmetric, we store only the unique D2 components
        const double axxx_3 = 3.0*U*axx_2;
        const double axxy_3 = 2.0*U*axy_2 + V*axx_2;
        const double axyy_3 = 2.0*V*axy_2 + U*ayy_2;
        const double ayyy_3 = 3.0*V*ayy_2;

      // equilibrium 2nd-order Hermite coefficient (athermal/isothermal):
      // a2_0,αβ = ρ (uα uβ + cs2 δαβ)
        const double axxxx_4 = 4.*U*axxx_3 - 6.*U2*axx_2;
        const double axxxy_4 = 3.*U*axxy_3 + V*axxx_3 - 3.*U2*axy_2 - 3.*U*V*axx_2;
        const double axyyy_4 = U*ayyy_3 + 3.*V*axyy_3 - 3.*U*V*ayy_2 - 3.*V2*axy_2;
        const double ayyyy_4 = 4.*V*ayyy_3 - 6.*V2*ayy_2;
        const double axxyy_4 = 2.*U*axyy_3 + 2.*V*axxy_3 - U2*ayy_2 - V2*axx_2 - 4.*U*V*axy_2;

        // --- Now loop over directions: build fneq,reg up to 4 and collide+stream
        for(int k=0; k<lattice.np; k++)
        {
            const double cxk = (double)lattice.cx[k];
            const double cyk = (double)lattice.cy[k];

            // Compact Hermite polynomials for D2 (raw, not normalized by cs)
            // H2:
            const double H2_xx = cxk*cxk - lattice.cs2;
            const double H2_yy = cyk*cyk - lattice.cs2;
            const double H2_xy = cxk*cyk;

            // H3:
            const double H3_xxx = cxk*cxk*cxk - 3.0*lattice.cs2*cxk;
            const double H3_xxy = cxk*cxk*cyk - lattice.cs2*cyk;
            const double H3_xyy = cxk*cyk*cyk - lattice.cs2*cxk;
            const double H3_yyy = cyk*cyk*cyk - 3.0*lattice.cs2*cyk;

            // H4:
            const double cx2 = cxk*cxk, cy2 = cyk*cyk;
            const double H4_xxxx = cx2*cx2 - 6.0*lattice.cs2*cx2 + 3.0*lattice.cs4;
            const double H4_xxxy = cx2*cxk*cyk - 3.0*lattice.cs2*cxk*cyk;
            const double H4_xxyy = cx2*cy2 - lattice.cs2*(cx2 + cy2) + lattice.cs4;
            const double H4_xyyy = cy2*cyk*cxk - 3.0*lattice.cs2*cxk*cyk;
            const double H4_yyyy = cy2*cy2 - 6.0*lattice.cs2*cy2 + 3.0*lattice.cs4;

            // Contractions a2:H2, a3:H3, a4:H4 with symmetry factors
            // a2 contraction: a_xx H_xx + a_yy H_yy + 2 a_xy H_xy
            const double c2 = axx_2*H2_xx + ayy_2*H2_yy + 2.0*axy_2*H2_xy;

            // a3 contraction: a_xxx H_xxx + 3 a_xxy H_xxy + 3 a_xyy H_xyy + a_yyy H_yyy
            const double c3 = axxx_3*H3_xxx + 3.0*axxy_3*H3_xxy + 3.0*axyy_3*H3_xyy + ayyy_3*H3_yyy;

            // a4 contraction: a_xxxx H_xxxx + 4 a_xxxy H_xxxy + 6 a_xxyy H_xxyy + 4 a_xyyy H_xyyy + a_yyyy H_yyyy
            const double c4 = axxxx_4*H4_xxxx + 4.0*axxxy_4*H4_xxxy + 6.0*axxyy_4*H4_xxyy
                            + 4.0*axyyy_4*H4_xyyy + ayyyy_4*H4_yyyy;

            // Regularized nonequilibrium up to order 4:
            const double fneq_reg = lattice.wf[k] * ( (c2)/(2.0*lattice.cs4) + (c3)/(6.0*lattice.cs6) + (c4)/(24.0*lattice.cs8) );

            // Regularized BGK collision:
            // f_post = feq + (1 - 1/tau)*fneq_reg = feq + omega1*fneq_reg
            f1(x, y, k) = f1(x, y, k) + parameters.omega1 * fneq_reg;
            double fluctuating_term = 0.;
            for(int h=3;h<lattice.np;h++)
                fluctuating_term += 1./lattice.b[h]*M(h,k)*etas[h]*sqrt(coeff*lattice.b[h]);
            f1(x,y,k) += lattice.wf[k]*fluctuating_term;
            int newx, newy;
            newx = x+lattice.cx[k];
            newy = y+lattice.cy[k];
            if(x==0 || x==parameters.nx-1)
              newx = (x+lattice.cx[k] + parameters.nx) % parameters.nx;
            if(y==0 || y==parameters.ny-1)
              newy = (y+lattice.cy[k] + parameters.ny) % parameters.ny;
            f2(newx,newy,k) = f1(x,y,k);
        }
    });
    Kokkos::fence(); 
}

// ----- run one simulation with given params, print stats periodically
static void run_single(const std::string& label, Params p)
{
    p.finalize();
    const D2Q9 lattice;

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "nx="<<p.nx<<" ny="<<p.ny
              << " rho0="<<p.rho0<<" kBT="<<p.kBT
              << " omega="<<p.omega
              << " T_ref="<<p.T_ref
              << " nsteps="<<p.nsteps
              << "\n";

    View3DArray f1("f1", p.nx,p.ny,lattice.np);
    View3DArray f2("f2", p.nx,p.ny,lattice.np);
    View2DArray rho("rho", p.nx,p.ny);
    View2DArray u("u", p.nx,p.ny);
    View2DArray v("v", p.nx,p.ny);
    View2DArray M("M", lattice.np, lattice.np);

    RNGPool rng_pool(123456789ULL);

    initial_state(rho,u,v,f1,f2,lattice,p, M);

    const double target = (p.rho0>0.0) ? (p.kBT/p.rho0) : 0.0;

    for(int it=0; it<=p.nsteps; ++it){
        algoLB(f1,f2,rho,u,v,lattice,p,rng_pool, M);
        std::swap(f1,f2);

        if(it % p.n_out == 0){
        auto s = compute_stats_simple(u,v,p);
        std::cout << std::setprecision(10)
          << "t/Tref="<<(it/p.T_ref)
          << "  <u>="<<s.mu<<" <v>="<<s.mv
          << "  <u^2>="<<s.u2<<" <v^2>="<<s.v2
          << "  target(kBT/rho)="<<target
          << "  max|u|="<<s.maxu<<" max|v|="<<s.maxv
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
    p.finalize();

    std::cout << "\n=== CASE 2: equipartition (kBT=1/3000) ===\n";

    const D2Q9 lattice;
    View3DArray f1("f1", p.nx,p.ny,lattice.np);
    View3DArray f2("f2", p.nx,p.ny,lattice.np);
    View2DArray rho("rho", p.nx,p.ny);
    View2DArray u("u", p.nx,p.ny);
    View2DArray v("v", p.nx,p.ny);
    View2DArray M("M", lattice.np, lattice.np);

    RNGPool rng_pool(123456789ULL);
    initial_state(rho,u,v,f1,f2,lattice,p,M);

    const int warmup = int(20.0*p.T_ref);
    const int nrun   = int(100.0*p.T_ref);

    double sum_u2 = 0.0;
    double sum_v2 = 0.0;
    long long nsamples = 0;

    for(int it=0; it<warmup; ++it){
        algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
        std::swap(f1,f2);
    }

    for(int it=0; it<nrun; ++it){
        algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
        std::swap(f1,f2);

        auto s = compute_stats_simple(u,v,p);
        sum_u2 += s.u2;
        sum_v2 += s.v2;
        nsamples++;
    }

    const double avg_u2 = sum_u2 / double(nsamples);
    const double avg_v2 = sum_v2 / double(nsamples);
    const double mean_t = 0.5*(avg_u2 + avg_v2);

    std::cout << std::setprecision(10)
              << "samples=" << nsamples
              << "  <u^2>_t=" << avg_u2
              << " <v^2>_t=" << avg_v2
              << "  mean_t=" << mean_t
              << "  target(kBT/rho0)=" << (p.kBT/p.rho0)
              << "\n";
}

static void case3_kbt_scaling(){
    std::cout << "\n=== CASE 3: kBT scaling (expect <u^2> ~ kBT/rho) ===\n";
    std::vector<double> kbts = {1.0/6000.0, 1.0/3000.0, 1.0/1500.0, 1.0/750.0};

    for(double kbt : kbts){
        Params p;
        p.kBT = kbt;
        p.finalize();

        const D2Q9 lattice;
        View3DArray f1("f1", p.nx,p.ny,lattice.np);
        View3DArray f2("f2", p.nx,p.ny,lattice.np);
        View2DArray rho("rho", p.nx,p.ny);
        View2DArray u("u", p.nx,p.ny);
        View2DArray v("v", p.nx,p.ny);
        View2DArray M("M", lattice.np, lattice.np);

        RNGPool rng_pool(123456789ULL);
        initial_state(rho,u,v,f1,f2,lattice,p,M);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        double sum_u2 = 0.0;
        double sum_v2 = 0.0;
        long long nsamples = 0;

        for(int it=0; it<warmup; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);
        }

        for(int it=0; it<nrun; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);

            auto s = compute_stats_simple(u,v,p);
            sum_u2 += s.u2;
            sum_v2 += s.v2;
            nsamples++;
        }

        const double avg_u2 = sum_u2 / double(nsamples);
        const double avg_v2 = sum_v2 / double(nsamples);
        const double mean_t = 0.5*(avg_u2 + avg_v2);

        std::cout << std::setprecision(10)
                  << "kBT=" << kbt
                  << "  samples=" << nsamples
                  << "  <u^2>_t=" << avg_u2
                  << " <v^2>_t=" << avg_v2
                  << "  mean_t=" << mean_t
                  << "  target=" << (kbt/p.rho0)
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

        const D2Q9 lattice;
        View3DArray f1("f1", p.nx,p.ny,lattice.np);
        View3DArray f2("f2", p.nx,p.ny,lattice.np);
        View2DArray rho("rho", p.nx,p.ny);
        View2DArray u("u", p.nx,p.ny);
        View2DArray v("v", p.nx,p.ny);
        View2DArray M("M", lattice.np, lattice.np);

        RNGPool rng_pool(123456789ULL);
        initial_state(rho,u,v,f1,f2,lattice,p,M);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        double sum_u2 = 0.0;
        double sum_v2 = 0.0;
        long long nsamples = 0;

        for(int it=0; it<warmup; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);
        }

        for(int it=0; it<nrun; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);

            auto s = compute_stats_simple(u,v,p);
            sum_u2 += s.u2;
            sum_v2 += s.v2;
            nsamples++;
        }

        const double avg_u2 = sum_u2 / double(nsamples);
        const double avg_v2 = sum_v2 / double(nsamples);
        const double mean_t = 0.5*(avg_u2 + avg_v2);

        std::cout << std::setprecision(10)
                  << "rho0=" << rho0
                  << "  samples=" << nsamples
                  << "  <u^2>_t=" << avg_u2
                  << " <v^2>_t=" << avg_v2
                  << "  mean_t=" << mean_t
                  << "  target=" << (p.kBT/rho0)
                  << "\n";
    }
}

static void case5_tau_sweep(){
    std::cout << "\n=== CASE 5: tau sweep via ni (equipartition at fixed kBT) ===\n";

    double taus[15] = {0.5, 0.5001, 0.5005, 0.501, 0.505, 0.51, 0.55, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0, 50.0, 100.0};
    const double kBT = 1.0/3000.0;

    for(double tau_target : taus){
        Params p;
        p.kBT = kBT;
        p.ni  = (tau_target - 0.5)/3.0;
        p.finalize();

        std::cout << std::setprecision(10)
                  << "\n-- target tau=" << tau_target
                  << "  set ni=" << p.ni
                  << "  actual tau=" << p.tau
                  << "  omega=" << p.omega
                  << "\n";

        const D2Q9 lattice;
        View3DArray f1("f1", p.nx,p.ny,lattice.np);
        View3DArray f2("f2", p.nx,p.ny,lattice.np);
        View2DArray rho("rho", p.nx,p.ny);
        View2DArray u("u", p.nx,p.ny);
        View2DArray v("v", p.nx,p.ny);
        View2DArray M("M", lattice.np, lattice.np);

        RNGPool rng_pool(123456789ULL);
        initial_state(rho,u,v,f1,f2,lattice,p,M);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        double sum_u2 = 0.0;
        double sum_v2 = 0.0;
        long long nsamples = 0;

        for(int it=0; it<warmup; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);
        }

        for(int it=0; it<nrun; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M);
            std::swap(f1,f2);

            auto s = compute_stats_simple(u,v,p);
            sum_u2 += s.u2;
            sum_v2 += s.v2;
            nsamples++;
        }

        const double avg_u2 = sum_u2 / double(nsamples);
        const double avg_v2 = sum_v2 / double(nsamples);
        const double mean_t = 0.5*(avg_u2 + avg_v2);

        std::cout << std::setprecision(10)
                  << "tau=" << p.tau
                  << "  samples=" << nsamples
                  << "  <u^2>_t=" << avg_u2
                  << " <v^2>_t=" << avg_v2
                  << "  mean_t=" << mean_t
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