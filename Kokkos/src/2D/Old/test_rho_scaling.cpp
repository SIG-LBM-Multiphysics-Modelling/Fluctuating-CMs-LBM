#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

    double rho0, kBT, U_ref, ni;
    double tau, omega, omega1, T_ref;
    int nsteps;

    Params(double rho0_, double kBT_, double U_ref_=0.01, double ni_=0.1)
    : rho0(rho0_), kBT(kBT_), U_ref(U_ref_), ni(ni_)
    {
        tau = ni*3. + 0.5;
        omega = 1./tau;
        omega1 = 1.-omega;
        T_ref = ny/U_ref;
        nsteps = 20000;
    }
};

static inline void initial_state(View2DArray rho, View2DArray u, View2DArray v,
                                 View3DArray f1, View3DArray f2,
                                 const D2Q9 lat, const Params p)
{
    Kokkos::parallel_for("init",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y)
    {
        const double R=p.rho0;
        rho(x,y)=R; u(x,y)=0.0; v(x,y)=0.0;
        for(int k=0;k<lat.np;k++){
            const double feq = lat.wf[k]*R;
            f1(x,y,k)=feq; f2(x,y,k)=feq;
        }
    });
    Kokkos::fence();
}

static inline void algoLB(View3DArray f1, View3DArray f2, View2DArray rho, View2DArray u, View2DArray v,
                          const D2Q9 lat, const Params p, RNGPool pool)
{
    Kokkos::parallel_for("lb",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y)
    {
        const double R =
            f1(x,y,0)+f1(x,y,1)+f1(x,y,2)+f1(x,y,3)+f1(x,y,4)+f1(x,y,5)+f1(x,y,6)+f1(x,y,7)+f1(x,y,8);

        double U = (f1(x,y,1) - f1(x,y,3) + f1(x,y,5) - f1(x,y,6) - f1(x,y,7) + f1(x,y,8))/R;
        double V = (f1(x,y,2) - f1(x,y,4) + f1(x,y,5) + f1(x,y,6) - f1(x,y,7) - f1(x,y,8))/R;

        rho(x,y)=R; u(x,y)=U; v(x,y)=V;

        auto rng = pool.get_state();
        const double eta3=rng.normal(), eta4=rng.normal(), eta5=rng.normal(), eta6=rng.normal(), eta7=rng.normal(), eta8=rng.normal();
        pool.free_state(rng);

        const double U2=U*U, V2=V*V;

        double r4 = f1(x,y,1)-f1(x,y,2)+f1(x,y,3)-f1(x,y,4);
        double r5 = f1(x,y,5)-f1(x,y,6)+f1(x,y,7)-f1(x,y,8);

        double k4 = r4 - R*(U2-V2);
        double k5 = r5 - R*(U*V);

        const double k3 = 2.*sqrt(R*p.kBT)*lat.cs*eta3;
        k4 = p.omega1*k4 + 2.*sqrt(R*p.kBT*p.omega*(2.-p.omega))*lat.cs*eta4;
        k5 = p.omega1*k5 +     sqrt(R*p.kBT*p.omega*(2.-p.omega))*lat.cs*eta5;
        const double k6 = sqrt(2.*R*p.kBT)*lat.cs2*eta6;
        const double k7 = sqrt(2.*R*p.kBT)*lat.cs2*eta7;
        const double k8 = 2.*sqrt(R*p.kBT)*lat.cs3*eta8;

        const double r1 = R*U;
        const double r2 = R*V;
        const double r3 = k3 + R*(U2+V2);
        r4 = k4 + R*(U2-V2);
        r5 = k5 + R*(U*V);

        const double r6 = k6 + 2.*U*r5 + 0.5*V*(k3+k4) + R*U2*V;
        const double r7 = k7 + 0.5*U*(k3-k4) + 2.*V*r5 + R*U*V2;
        const double r8 = k8 + 2.*U*r7 + 2.*V*r6 + 0.5*k3*(U2+V2) - 0.5*k4*(U2-V2)
                        + R*U2*V2 + 4.*U*V*r5;

        f1(x,y,0) = 4.*R/9. - 2.*r3/3. + r8;
        f1(x,y,1) = r4/4. - (r7 + r8)/2. + r3/12. + R/9. + r1/3.;
        f1(x,y,2) = r3/12. - r4/4. - (r6 + r8)/2. + R/9. + r2/3.;
        f1(x,y,3) = r7/2. - r8/2. + r4/4. + r3/12. + R/9. - r1/3.;
        f1(x,y,4) = r6/2. - r8/2. - r4/4. + r3/12. + R/9. - r2/3.;
        f1(x,y,5) = (r5 + r6 + r7 + r8)/4. + (r1 + r2 + r3)/12. + R/36.;
        f1(x,y,6) = (r6 + r8)/4. - (r5 + r7)/4. + (r2 + r3)/12. - r1/12. + R/36.;
        f1(x,y,7) = (r5 + r8)/4. - (r6 + r7)/4. + r3/12. - (r1 + r2)/12. + R/36.;
        f1(x,y,8) = (r7 + r8)/4. - (r5 + r6)/4. + (r1 + r3)/12. - r2/12. + R/36.;

        for(int k=0;k<lat.np;k++){
            const int nx2 = (x + lat.cx[k] + p.nx) % p.nx;
            const int ny2 = (y + lat.cy[k] + p.ny) % p.ny;
            f2(nx2,ny2,k)=f1(x,y,k);
        }
    });
    Kokkos::fence();
}

static inline void stats_u2v2(View2DArray u, View2DArray v, const Params p, double& u2, double& v2)
{
    double sum_u2=0.0, sum_v2=0.0;

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

    const double N = double(p.nx)*double(p.ny);
    u2 = sum_u2 / N;
    v2 = sum_v2 / N;
}

int main(int argc, char* argv[])
{
    std::string outdir="./TEST_rho_scaling";
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i], "-d")==0 && i+1<argc) outdir=argv[++i];
    }
    std::filesystem::create_directories(outdir);
    const std::string outpath = outdir + "/rho_scaling_summary.txt";

    std::ofstream out(outpath, std::ios::trunc);
    out << "# rho0   target(kBT/rho0)   measured<u^2>   measured<v^2>\n";

    const D2Q9 lat;

    Kokkos::initialize(argc, argv);
    {
        const double kBT = 1.0/3000.0;
        const double rho_list[3] = { 0.5, 1.0, 2.0 };

        for(int c=0;c<3;c++){
            const Params p(rho_list[c], kBT);

            View3DArray f1("f1", p.nx, p.ny, lat.np);
            View3DArray f2("f2", p.nx, p.ny, lat.np);
            View2DArray rho("rho", p.nx, p.ny);
            View2DArray u("u", p.nx, p.ny);
            View2DArray v("v", p.nx, p.ny);

            RNGPool pool(123456789ULL);
            initial_state(rho,u,v,f1,f2,lat,p);

            for(int it=0; it<p.nsteps; ++it){
                algoLB(f1,f2,rho,u,v,lat,p,pool);
                std::swap(f1,f2);
            }

            double u2, v2;
            stats_u2v2(u,v,p,u2,v2);
            const double target = p.kBT/p.rho0;

            std::cout << std::setprecision(10)
                      << "rho0=" << p.rho0
                      << "  target=" << target
                      << "  <u^2>=" << u2
                      << "  <v^2>=" << v2 << "\n";

            out << std::setprecision(16)
                << p.rho0 << "\t" << target << "\t" << u2 << "\t" << v2 << "\n";
        }
    }
    Kokkos::finalize();

    std::cout << "Wrote: " << outpath << "\n";
    return 0;
}
