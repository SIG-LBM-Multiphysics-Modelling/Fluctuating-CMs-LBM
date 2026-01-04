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
    static constexpr int dim = 2;
    static constexpr int np = 9;
    const int cx[np];
    const int cy[np];
    const int opp[np];
    const double wf[np];
    const double b[np];

    D2Q9():
         b{1., 1./3., 1./3., 4./9., 4./9., 4./9., 1./9., 2./27., 2./27., 4./81.}
        cx{0, 1, 0, -1,  0, 1, -1, -1,  1},
        cy{0, 0, 1,  0, -1, 1,  1, -1, -1},
       opp{0, 3, 4,  1,  2, 7,  8,  5,  6},
        // Weight factors for lattice directions
        wf{4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.} {}
};

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
    Id(0,0) = Id(1,1) = lattice.cs2;
    Id(0,1) = Id(1,0) = 0.;
    double hat_[2];
    for(int k=0; k<lattice.np; k++)
    {
      hat_[0] = lattice.cx[k];
      hat_[1] = lattice.cy[k];
      for(int a=0; a<2; a++)
      {
        H1(a) = hat_a; 
        for(int b=0; b<2; b++)
        {
          H2(a,b) = hat[a]*hat_[b]-Id(a,b);
          for(int c=0; c<2; c++)
          {
            H3(a,b,c) = hat_[a]*hat_[b]*hat_[c]-hat_[a]*Id(b,c)+hat_[a]*Id(a,c)+hat_[a]*Id(a,b);
            for(int d=0; d<2; d++)
            {
              double hat_II = hat_[a]*hat_[b]*Id(c,d)+hat_[a]*hat_[c]*Id(b,d)+
                              hat_[a]*hat_[d]*Id(b,c)+hat_[b]*hat_[c]*Id(a,d)+
                              hat_[b]*hat_[d]*Id(a,c)+hat_[c]*hat_[d]*Id(a,b);
              double II = Id(a,b)*Id(c,d)+Id(a,c)*Id(b,d)+Id(a,d)*Id(b,c);
              H4(a,b,c,d) = hat_[a]*hat_[b]*hat_[c]*hat_[d]-hat_II+II;
            }
          }
        }
      }
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

        a_eq_0 = R;
        a_eq_1x = a_eq_0*U;
        a_eq_1y = a_eq_0*V;
        a_eq_2xx = a_eq_1x*U;
        a_eq_2yy = a_eq_1y*V;
        a_eq_2xy = a_eq_1x*V;
        a_eq_2yx = a_eq_1y*U;
        a_eq_3xxx = a_eq_2xx*U;
        a_eq_3yxx = a_eq_2yx*U;
        a_eq_3xyx = a_eq_2xy*U;
        a_eq_3xxy = a_eq_2xx*V;
        a_eq_3xyy = a_eq_2xy*V;
        a_eq_3yyx = a_eq_2yy*U;
        a_eq_3yxy = a_eq_2yx*V;
        a_eq_3yyy = a_eq_2yy*V;
        a_eq_4xxxx = a_eq_3xxx*U;
        a_eq_4xxxy = a_eq_3xxx*V;
        a_eq_4xxyx = a_eq_3xxy*U;
        a_eq_4xyxx = a_eq_3xyx*U;
        a_eq_4yxxx = a_eq_3yxx*U;
        
        a_eq_4yyxx = a_eq_3yyx*U;
        a_eq_4xyyx = a_eq_3xyy*U;
        a_eq_4xxyy = a_eq_3xxy*V;
        a_eq_4yxyx = a_eq_3yxy*U;
        a_eq_4xyxy = a_eq_3xyx*V;
        
        a_eq_4xyyy = a_eq_3xyy*V;
        a_eq_4yxyy = a_eq_3yxy*V;
        a_eq_4yyxy = a_eq_3yyx*V;
        a_eq_4yyyx = a_eq_3yyy*U;
        
        a_eq_4yyyy = a_eq_3yyy*V;

        a_neq_0 = 0.;
        a_neq_1x = a_eq_0*U;
        a_neq_1y = a_eq_0*V;

        a_neq_2xx = a_neq_2xy = a_neq_2yx = a_neq_2yy = 0.;
        a_neq_3xxx = a_neq_3yxx = a_neq_3xyx = a_neq_3xxy = a_neq_3xyy = a_neq_3yyx = a_neq_3yxy = a_neq_3yyy = 0.;
        for(int k=0; k<lattice.np; k++)
        {
            const double cu = U*lattice.cx[l] + V*lattice.cy[l];
            const double feq = lattice.wf[l] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(U*U+V*V));
            const double fneq = f1(x,y,l)-feq;
            a_neq_2xx += (lattice.cx[k]*lattice.cx[k]-lattice.cs2)*fneq;
            a_neq_2xy += lattice.cx[k]*lattice.cy[k]*fneq;
            a_neq_2yx += lattice.cy[k]*lattice.cx[k]*fneq;
            a_neq_2yy += (lattice.cy[k]*lattice.cy[k]-lattice.cs2)*fneq;

            a_neq_3xxx += H3(x,x,x)*fneq;
            a_neq_3yxx += H3(y,x,x)*fneq;
            a_neq_3xyx += H3(x,y,x)*fneq;
            a_neq_3xxy += H3(x,x,y)*fneq;
            a_neq_3xyy += H3(x,y,y)*fneq;
            a_neq_3yyx += H3(y,y,x)*fneq;
            a_neq_3yxy += H3(y,x,y)*fneq;
            a_neq_3yyy += H3(y,y,y)*fneq;

            a_neq_4xxxx = H4(x,x,x,x)*fneq;
            a_neq_4xxxy = H4(x,x,x,y)*fneq;
            a_neq_4xxyx = H4(x,x,y,x)*fneq;
            a_neq_4xyxx = H4(x,y,x,x)*fneq;
            a_neq_4yxxx = H4(y,x,x,x)*fneq;
            
            a_neq_4yyxx = H4(y,y,x,x)*fneq;
            a_neq_4xyyx = H4(x,y,y,x)*fneq;
            a_neq_4xxyy = H4(x,x,y,y)*fneq;
            a_neq_4yxyx = H4(y,x,y,x)*fneq;
            a_neq_4xyxy = H4(x,y,x,y)*fneq;
            
            a_neq_4xyyy = H4(x,y,y,y)*fneq;
            a_neq_4yxyy = H4(y,x,y,y)*fneq;
            a_neq_4yyxy = H4(y,y,x,y)*fneq;
            a_neq_4yyyx = H4(y,y,y,x)*fneq;
            
            a_neq_4yyyy = H4(y,y,y,y)*fneq;
        }

        for(int l=0;l<lattice.np;l++){
          const double cu = U*lattice.cx[l] + V*lattice.cy[l];
          const double feq = lattice.wf[l] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*(U*U+V*V));
          f1(x,y,l) = feq + parameters.omega1*(f1(x,y,l)-feq);
                
          double fluctuating_term = 0.;
          for(int k=4; k<lattice.np; k++)
            fluctuating_term += 1./lattice.b[k]*M(k,l)*etas[k]*sqrt(coeff*lattice.b[k]);

          f1(x,y,z,l) += lattice.wf[l]*fluctuating_term;
          const int newx = (x + lattice.cx[l] + parameters.nx) % parameters.nx;
          const int newy = (y + lattice.cy[l] + parameters.ny) % parameters.ny;
          f2(newx,newy,l) = f1(x,y,l);
        }
    });
    Kokkos::fence(); 
}

// ----- run one simulation with given params, print stats periodically
static void run_single(const std::string& label, Params p)
{
    p.finalize();

    std::cout << "\n=== " << label << " ===\n";
    std::cout << "nx="<<p.nx<<" ny="<<p.ny
              << " rho0="<<p.rho0<<" kBT="<<p.kBT
              << " omega="<<p.omega
              << " T_ref="<<p.T_ref
              << " nsteps="<<p.nsteps
              << "\n";

    const D2Q9 lattice;
    View3DArray f1("f1", p.nx,p.ny,lattice.np);
    View3DArray f2("f2", p.nx,p.ny,lattice.np);
    View1DArray rho("rho", p.nx,p.ny);
    View1DArray u("u", p.nx,p.ny);
    View1DArray v("v", p.nx,p.ny);
    View2DArray M("M", 9, 9);

    View2DArray Id("Id", 2, 2);
    View1DArray H1("H1", 2);
    View2DArray H2("H2", 2, 2);
    View3DArray H3("H3", 2, 2, 2);
    View4DArray H4("H4", 2, 2, 2, 2);

    RNGPool rng_pool(123456789ULL);

    initial_state(rho,u,v,f1,f2,lattice,p, M, H1, H2, H3, H4, Id);

    const double target = (p.rho0>0.0) ? (p.kBT/p.rho0) : 0.0;

    for(int it=0; it<=p.nsteps; ++it){
        algoLB(f1,f2,rho,u,v,lattice,p,rng_pool, M, H1, H2, H3, H4, Id);
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
    run_single("CASE 2: equipartition (kBT=1/3000)", p);
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
        View1DArray rho("rho", p.nx,p.ny);
        View1DArray u("u", p.nx,p.ny);
        View1DArray v("v", p.nx,p.ny);
        View2DArray M("M", 9, 9);
  
        View2DArray Id("Id", 2, 2);
        View1DArray H1("H1", 2);
        View2DArray H2("H2", 2, 2);
        View3DArray H3("H3", 2, 2, 2);
        View4DArray H4("H4", 2, 2, 2, 2);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,f1,f2,lattice,p,M, H1, H2, H3, H4, Id);

        // warmup shorter to keep it fast
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool,M, H1, H2, H3, H4, Id);
            std::swap(f1,f2);
        }

        auto s = compute_stats_simple(u,v,p);
        const double N = double(p.nx)*p.ny;
        const double u2 = s.u2, v2 = s.v2;
        const double mean = (u2+v2)/3.0;

        std::cout << std::setprecision(10)
          << "kBT="<<kbt
          << "  <u^2>="<<u2<<" <v^2>="<<v2
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

        const D2Q9 lattice;
        View3DArray f1("f1", p.nx,p.ny,lattice.np);
        View3DArray f2("f2", p.nx,p.ny,lattice.np);
        View1DArray rho("rho", p.nx,p.ny);
        View1DArray u("u", p.nx,p.ny);
        View1DArray v("v", p.nx,p.ny);
        View2DArray M("M", 9, 9);

        VView2DArray Id("Id", 2, 2);
        View1DArray H1("H1", 2);
        View2DArray H2("H2", 2, 2);
        View3DArray H3("H3", 2, 2, 2);
        View4DArray H4("H4", 2, 2, 2, 2);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,f1,f2,lattice,p,M, H1, H2, H3, H4, Id);

        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,attice,p,rng_pool, M, H1, H2, H3, H4, Id);
            std::swap(f1,f2);
        }

        auto s = compute_stats_simple(u,v,p);
        const double N = double(p.nx)*p.ny;
        const double u2 = s.u2, v2 = s.v2;
        const double mean = (u2+v2)/3.0;

        std::cout << std::setprecision(10)
          << "rho0="<<rho0
          << "  <u^2>="<<u2<<" <v^2>="<<v2
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

        const D2Q9 lattice;
        View3DArray f1("f1", p.nx,p.ny,lattice.np);
        View3DArray f2("f2", p.nx,p.ny,lattice.np);
        View1DArray rho("rho", p.nx,p.ny);
        View1DArray u("u", p.nx,p.ny);
        View1DArray v("v", p.nx,p.ny);
        View2DArray M("M", 9, 9);

        View2DArray Id("Id", 2, 2);
        View1DArray H1("H1", 2);
        View2DArray H2("H2", 2, 2);
        View3DArray H3("H3", 2, 2, 2);
        View4DArray H4("H4", 2, 2, 2, 2);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,f1,f2,lattice,p, M, H1, H2, H3, H4, Id);

        // same warmup/run idea as your scaling cases
        const int warmup = int(20.0*p.T_ref);
        const int nrun   = int(100.0*p.T_ref);

        for(int it=0; it<warmup+nrun; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool, M, H1, H2, H3, H4, Id);
            std::swap(f1,f2);
        }

        // report the same stats you already trust
        auto s = compute_stats_simple(u,v,p);
        const double mean_u2 = (s.u2 + s.v2)/3.0;

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