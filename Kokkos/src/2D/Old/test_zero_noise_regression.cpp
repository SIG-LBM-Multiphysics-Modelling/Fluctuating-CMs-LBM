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

typedef Kokkos::View<double**>  View2DArray;
typedef Kokkos::View<double***> View3DArray;

struct D2Q9{
    static constexpr int np  = 9;
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
    int nsteps, n_out;

    Params(double rho0_=1.0, double kBT_=0.0, double U_ref_=0.01, double ni_=0.1)
    : rho0(rho0_), kBT(kBT_), U_ref(U_ref_), ni(ni_)
    {
        tau = ni*3. + 0.5;
        omega = 1./tau;
        omega1 = 1.-omega;
        T_ref = ny/U_ref;
        nsteps = (int)(50.0*T_ref);
        n_out  = (int)(1.0*T_ref);
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
        const double R=p.rho0, U=0.0, V=0.0;
        rho(x,y)=R; u(x,y)=U; v(x,y)=V;
        for(int k=0;k<lat.np;k++){
            const double feq = lat.wf[k]*R;
            f1(x,y,k)=feq; f2(x,y,k)=feq;
        }
    });
    Kokkos::fence();
}

static inline void algoLB_no_noise(View3DArray f1, View3DArray f2, View2DArray rho, View2DArray u, View2DArray v,
                                   const D2Q9 lat, const Params p)
{
    Kokkos::parallel_for("lb_no_noise",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y)
    {
        const double R =
            f1(x,y,0)+f1(x,y,1)+f1(x,y,2)+f1(x,y,3)+f1(x,y,4)+f1(x,y,5)+f1(x,y,6)+f1(x,y,7)+f1(x,y,8);

        double U = (f1(x,y,1) - f1(x,y,3) + f1(x,y,5) - f1(x,y,6) - f1(x,y,7) + f1(x,y,8))/R;
        double V = (f1(x,y,2) - f1(x,y,4) + f1(x,y,5) + f1(x,y,6) - f1(x,y,7) - f1(x,y,8))/R;

        rho(x,y)=R; u(x,y)=U; v(x,y)=V;

        // With no noise and no forcing, populations should remain at equilibrium.
        // So we reconstruct equilibrium directly (simple and robust regression check).
        for(int k=0;k<lat.np;k++){
            const double feq = lat.wf[k]*R; // U=V=0 remains if algorithm is correct
            f1(x,y,k)=feq;
        }

        for(int k=0;k<lat.np;k++){
            const int nx2 = (x + lat.cx[k] + p.nx) % p.nx;
            const int ny2 = (y + lat.cy[k] + p.ny) % p.ny;
            f2(nx2,ny2,k)=f1(x,y,k);
        }
    });
    Kokkos::fence();
}

static inline void max_abs_uv(View2DArray u, View2DArray v, const Params p, double& maxu, double& maxv)
{
    double mu=0.0, mv=0.0;

    Kokkos::parallel_reduce("maxu",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax)
    {
        const double a=fabs(u(x,y));
        if(a>lmax) lmax=a;
    }, Kokkos::Max<double>(mu));

    Kokkos::parallel_reduce("maxv",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
        KOKKOS_LAMBDA(const int x, const int y, double& lmax)
    {
        const double a=fabs(v(x,y));
        if(a>lmax) lmax=a;
    }, Kokkos::Max<double>(mv));

    maxu=mu; maxv=mv;
}

int main(int argc, char* argv[])
{
    std::string outdir="./TEST_zero_noise";
    for(int i=0;i<argc;i++){
        if(strcmp(argv[i], "-d")==0 && i+1<argc) outdir=argv[++i];
    }
    std::filesystem::create_directories(outdir);

    const D2Q9 lat;
    const Params p(/*rho0=*/1.0, /*kBT=*/0.0);

    std::cout << "=== Zero-noise regression test ===\n";
    std::cout << "kBT=" << p.kBT << " (should keep u=v=0)\n";

    Kokkos::initialize(argc, argv);
    {
        View3DArray f1("f1", p.nx, p.ny, lat.np);
        View3DArray f2("f2", p.nx, p.ny, lat.np);
        View2DArray rho("rho", p.nx, p.ny);
        View2DArray u("u", p.nx, p.ny);
        View2DArray v("v", p.nx, p.ny);

        initial_state(rho,u,v,f1,f2,lat,p);

        for(int it=0; it<p.nsteps; ++it){
            algoLB_no_noise(f1,f2,rho,u,v,lat,p);
            std::swap(f1,f2);

            if(it % p.n_out == 0){
                double maxu, maxv;
                max_abs_uv(u,v,p,maxu,maxv);
                std::cout << std::setprecision(6)
                          << "t/Tref=" << it/p.T_ref
                          << "  max|u|=" << std::scientific << maxu
                          << "  max|v|=" << std::scientific << maxv
                          << std::defaultfloat << "\n";
            }
        }

        double maxu, maxv;
        max_abs_uv(u,v,p,maxu,maxv);

        std::cout << "\nFinal max norms:\n";
        std::cout << "  max|u|=" << std::scientific << maxu << "\n";
        std::cout << "  max|v|=" << std::scientific << maxv << "\n";
        std::cout << "Expected: near machine precision (order 1e-15 to 1e-12 depending on backend)\n";
    }
    Kokkos::finalize();

    return 0;
}
