#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using namespace std;
using RNGPool = Kokkos::Random_XorShift64_Pool<>;

typedef Kokkos::View<double**>  View2DArray;
typedef Kokkos::View<double***> View3DArray;

struct D2Q9{
    static constexpr double cs  = 0.57735026918962576451;
    static constexpr double cs2 = cs*cs;
    static constexpr double cs3 = cs2*cs;
    static constexpr int np = 9;

    const int cx[np];
    const int cy[np];
    const double wf[np];

    D2Q9():
        cx{0, 1, 0, -1,  0, 1, -1, -1,  1},
        cy{0, 0, 1,  0, -1, 1,  1, -1, -1},
        wf{4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.} {}
};

struct Params{
    static constexpr int nx = 200;
    static constexpr int ny = nx/2;

    double rho0;
    double kBT;
    double U_ref;
    double ni;

    double tau;
    double omega;
    double omega1;
    double T_ref;

    int nsteps;
    int n_out;

    Params(double rho0_, double kBT_, double U_ref_=0.01, double ni_=0.1)
    : rho0(rho0_), kBT(kBT_), U_ref(U_ref_), ni(ni_)
    {
        tau   = ni*3. + 0.5;
        omega = 1./tau;
        omega1= 1. - omega;
        T_ref = ny / U_ref;
        nsteps = (int)(100.0 * T_ref);
        n_out  = (int)(1.0   * T_ref);
    }
};

static inline void write_header(const std::string& path){
    std::ofstream out(path, std::ios::trunc);
    out << "# t/T_ref   u_center/U_ref   v_center/U_ref   <u>   <v>   <u^2>   <v^2>   max|u|   max|v|\n";
}

static inline void append_line(const std::string& path,
                               double t_over_Tref,
                               double uc_over_Uref,
                               double vc_over_Uref,
                               double mean_u, double mean_v,
                               double mean_u2, double mean_v2,
                               double maxabs_u, double maxabs_v)
{
    std::ofstream out(path, std::ios::app);
    out << std::setprecision(16)
        << t_over_Tref << "\t"
        << uc_over_Uref << "\t"
        << vc_over_Uref << "\t"
        << mean_u << "\t"
        << mean_v << "\t"
        << mean_u2 << "\t"
        << mean_v2 << "\t"
        << maxabs_u << "\t"
        << maxabs_v << "\n";
}

void initial_state(View2DArray rho, View2DArray u, View2DArray v,
                   View3DArray f1, View3DArray f2,
                   const D2Q9 lattice, const Params p)
{
    Kokkos::parallel_for("InitializeState",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y)
    {
        const double R = p.rho0;
        const double U = 0.0;
        const double V = 0.0;

        rho(x,y) = R;
        u(x,y)   = U;
        v(x,y)   = V;

        for(int k=0;k<lattice.np;k++){
            const double first_order = U*lattice.cx[k] + V*lattice.cy[k];
            const double feq = lattice.wf[k]*R*(1. + 3.*first_order + 4.5*first_order*first_order - 1.5*(U*U+V*V));
            f1(x,y,k) = feq;
            f2(x,y,k) = feq;
        }
    });
    Kokkos::fence();
}

void algoLB(View3DArray f1, View3DArray f2, View2DArray rho, View2DArray u, View2DArray v,
            const D2Q9 lattice, const Params p, RNGPool rng_pool)
{
    Kokkos::parallel_for("LBAlgorithm",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y)
    {
        const double R =
            f1(x,y,0)+f1(x,y,1)+f1(x,y,2)+f1(x,y,3)+f1(x,y,4)+f1(x,y,5)+f1(x,y,6)+f1(x,y,7)+f1(x,y,8);

        double U = (f1(x,y,1) - f1(x,y,3) + f1(x,y,5) - f1(x,y,6) - f1(x,y,7) + f1(x,y,8))/R;
        double V = (f1(x,y,2) - f1(x,y,4) + f1(x,y,5) + f1(x,y,6) - f1(x,y,7) - f1(x,y,8))/R;

        // no forcing
        const double Fx = 0.0;
        const double Fy = 0.0;
        U += 0.5*Fx;
        V += 0.5*Fy;

        rho(x,y) = R;
        u(x,y) = U;
        v(x,y) = V;

        const double U2 = U*U;
        const double V2 = V*V;

        auto rng = rng_pool.get_state();
        const double eta3 = rng.normal();
        const double eta4 = rng.normal();
        const double eta5 = rng.normal();
        const double eta6 = rng.normal();
        const double eta7 = rng.normal();
        const double eta8 = rng.normal();
        rng_pool.free_state(rng);

        double r4 = f1(x,y,1)-f1(x,y,2)+f1(x,y,3)-f1(x,y,4);
        double r5 = f1(x,y,5)-f1(x,y,6)+f1(x,y,7)-f1(x,y,8);

        double k4 = r4 - R*(U2 - V2);
        double k5 = r5 - R*(U*V);

        const double k1 = 0.5*Fx;
        const double k2 = 0.5*Fy;

        const double k3 = 2.*sqrt(R*p.kBT)*lattice.cs*eta3;
        k4 = p.omega1*k4 + 2.*sqrt(R*p.kBT*p.omega*(2.-p.omega))*lattice.cs*eta4;
        k5 = p.omega1*k5 +     sqrt(R*p.kBT*p.omega*(2.-p.omega))*lattice.cs*eta5;
        const double k6 = 0.5*Fy*lattice.cs2 + sqrt(2.*R*p.kBT)*lattice.cs2*eta6;
        const double k7 = 0.5*Fx*lattice.cs2 + sqrt(2.*R*p.kBT)*lattice.cs2*eta7;
        const double k8 = 2.*sqrt(R*p.kBT)*lattice.cs3*eta8;

        const double r1 = k1 + R*U;
        const double r2 = k2 + R*V;
        const double r3 = k3 + 2.*U*k1 + 2.*V*k2 + R*(U2+V2);
        r4 = k4 + 2.*U*k1 - 2.*V*k2 + R*(U2-V2);
        r5 = k5 + U*k2 + V*k1 + R*(U*V);

        const double r6 = k6 + 2.*U*r5 + 0.5*V*(k3+k4) + U2*k2 + 2.*U*V*k1 + R*U2*V;
        const double r7 = k7 + 0.5*U*(k3-k4) + 2.*V*r5 + V2*k1 + 2.*U*V*k2 + R*U*V2;
        const double r8 = k8 + 2.*U*r7 + 2.*V*r6 + 0.5*k3*(U2+V2) - 0.5*k4*(U2-V2)
                        + R*U2*V2 + 4.*U*V*r5 + 2.*U*V2*k1 + 2.*U2*V*k2;

        f1(x,y,0) = 4.*R/9. - 2.*r3/3. + r8;
        f1(x,y,1) = r4/4. - (r7 + r8)/2. + r3/12. + R/9. + r1/3.;
        f1(x,y,2) = r3/12. - r4/4. - (r6 + r8)/2. + R/9. + r2/3.;
        f1(x,y,3) = r7/2. - r8/2. + r4/4. + r3/12. + R/9. - r1/3.;
        f1(x,y,4) = r6/2. - r8/2. - r4/4. + r3/12. + R/9. - r2/3.;
        f1(x,y,5) = (r5 + r6 + r7 + r8)/4. + (r1 + r2 + r3)/12. + R/36.;
        f1(x,y,6) = (r6 + r8)/4. - (r5 + r7)/4. + (r2 + r3)/12. - r1/12. + R/36.;
        f1(x,y,7) = (r5 + r8)/4. - (r6 + r7)/4. + r3/12. - (r1 + r2)/12. + R/36.;
        f1(x,y,8) = (r7 + r8)/4. - (r5 + r6)/4. + (r1 + r3)/12. - r2/12. + R/36.;

        for(int k=0;k<lattice.np;k++){
            const int newx = (x + lattice.cx[k] + p.nx) % p.nx;
            const int newy = (y + lattice.cy[k] + p.ny) % p.ny;
            f2(newx,newy,k) = f1(x,y,k);
        }
    });
    Kokkos::fence();
}

static inline void compute_stats(View2DArray u, View2DArray v,
                                 const Params p,
                                 double& mean_u, double& mean_v,
                                 double& mean_u2, double& mean_v2,
                                 double& maxabs_u, double& maxabs_v)
{
    double sum_u=0.0, sum_v=0.0, sum_u2=0.0, sum_v2=0.0;

    Kokkos::parallel_reduce("sum_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            lsum += u(x,y);
        }, sum_u);

    Kokkos::parallel_reduce("sum_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            lsum += v(x,y);
        }, sum_v);

    Kokkos::parallel_reduce("sum_u2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            const double uu = u(x,y);
            lsum += uu*uu;
        }, sum_u2);

    Kokkos::parallel_reduce("sum_v2",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lsum){
            const double vv = v(x,y);
            lsum += vv*vv;
        }, sum_v2);

    double mu=0.0, mv=0.0;
    Kokkos::parallel_reduce("max_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax){
            const double a = fabs(u(x,y));
            if(a > lmax) lmax = a;
        }, Kokkos::Max<double>(mu));

    Kokkos::parallel_reduce("max_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax){
            const double a = fabs(v(x,y));
            if(a > lmax) lmax = a;
        }, Kokkos::Max<double>(mv));

    const double N = double(p.nx)*double(p.ny);
    mean_u  = sum_u / N;
    mean_v  = sum_v / N;
    mean_u2 = sum_u2 / N;
    mean_v2 = sum_v2 / N;
    maxabs_u = mu;
    maxabs_v = mv;
}

int main(int argc, char* argv[])
{
    std::string outdir = "./TEST_equilibrium";
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i], "-d")==0 && i+1<argc) outdir = argv[++i];
    }
    std::filesystem::create_directories(outdir);
    const std::string data_path = outdir + "/equilibrium_stats.txt";

    const D2Q9 lattice;
    const Params p(/*rho0=*/1.0, /*kBT=*/1.0/3000.0, /*U_ref=*/0.01, /*ni=*/0.1);
    const double target_var = p.kBT / p.rho0;

    std::cout << "=== Equilibrium equipartition test ===\n";
    std::cout << "Target <u^2>=<v^2> ~ kBT/rho0 = " << std::setprecision(16) << target_var << "\n";
    std::cout << "Writing: " << data_path << "\n";

    write_header(data_path);

    Kokkos::initialize(argc, argv);
    {
        View3DArray f1("f1", p.nx, p.ny, lattice.np);
        View3DArray f2("f2", p.nx, p.ny, lattice.np);
        View2DArray rho("rho", p.nx, p.ny);
        View2DArray u("u", p.nx, p.ny);
        View2DArray v("v", p.nx, p.ny);

        RNGPool rng_pool(123456789ULL);

        initial_state(rho,u,v,f1,f2,lattice,p);

        for(int it=0; it<p.nsteps; ++it){
            algoLB(f1,f2,rho,u,v,lattice,p,rng_pool);
            std::swap(f1,f2);

            if(it % p.n_out == 0){
                double mean_u, mean_v, mean_u2, mean_v2, maxu, maxv;
                compute_stats(u,v,p, mean_u,mean_v,mean_u2,mean_v2, maxu,maxv);

                const int xc = p.nx/2, yc = p.ny/2;
                const double uc = u(xc,yc), vc = v(xc,yc);

                append_line(data_path, it/p.T_ref, uc/p.U_ref, vc/p.U_ref,
                            mean_u, mean_v, mean_u2, mean_v2, maxu, maxv);

                std::cout << std::setprecision(6)
                          << "t/Tref=" << it/p.T_ref
                          << "  <u^2>=" << mean_u2
                          << "  <v^2>=" << mean_v2
                          << "  target=" << target_var
                          << "  max|u|=" << maxu
                          << "  max|v|=" << maxv
                          << "\n";
            }
        }
    }
    Kokkos::finalize();

    return 0;
}

