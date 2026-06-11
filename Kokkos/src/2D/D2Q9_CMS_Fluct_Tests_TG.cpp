#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include "VTKWriter2D.h"
#include <Kokkos_Random.hpp>

using namespace std;
using RNGPool = Kokkos::Random_XorShift64_Pool<>;

typedef Kokkos::View<double***> View3DArray;
typedef Kokkos::View<double**> View2DArray;

struct D2Q9 {
  static constexpr double cs = 0.57735026918962576451;
  static constexpr double cs2 = cs*cs;
  static constexpr int dim = 2;
  static constexpr int np = 9;
  const int cx[np];
  const int cy[np];
  const int opp[np];
  const double wf[np];
  const double b[np];

  D2Q9():
    cx{0,1,0,-1,0,1,-1,-1,1},
    cy{0,0,1,0,-1,1,1,-1,-1},
    opp{0,3,4,1,2,7,8,5,6},
    wf{4./9.,1./9.,1./9.,1./9.,1./9.,1./36.,1./36.,1./36.,1./36.},
    b{1., 1./3., 1./3., 4./9., 4./9., 1./9., 2./27., 2./27., 4./81.}
  {}
};

struct Params {
  double rho0 = 1.0;
  double U0 = 0.02;
  int nx = 128;
  int ny = 128;
  double ni = 1e-8;
  double tau = ni*3.0 + 0.5;
  double omega = 1.0/tau;
  double omega1 = 1.0 - omega;
  double kBT = 0;
  int nsteps = 5000;
  int n_out = 100;
  double Lx = 2.0*M_PI;
  double Ly = 2.0*M_PI;
  double dx = 1.0;
  double dy = 1.0;

  void finalize() {
    tau = ni*3.0 + 0.5;
    omega = 1.0/tau;
    omega1 = 1.0 - omega;
    dx = Lx / double(nx);
    dy = Ly / double(ny);
  }
};

struct StatsSimple {
  double mu=0, mv=0;
  double u2=0, v2=0, e=0;
  double maxu=0, maxv=0, maxspd=0;
  double vort2=0;
};

struct TGModeStats {
  double Au=0, Av=0, A=0;
  double Eu=0, Ev=0;
};

static TGModeStats compute_tg_mode(const View2DArray& u, const View2DArray& v, const Params& p) {
  const double N = double(p.nx) * double(p.ny);
  double su=0.0, sv=0.0, eu=0.0, ev=0.0;
  Kokkos::parallel_reduce("tg_proj_u", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){
      const double x = (i + 0.5) * p.dx;
      const double y = (j + 0.5) * p.dy;
      acc += u(i,j) * sin(x) * cos(y);
    }, Kokkos::Sum<double>(su));
  Kokkos::parallel_reduce("tg_proj_v", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){
      const double x = (i + 0.5) * p.dx;
      const double y = (j + 0.5) * p.dy;
      acc += -v(i,j) * cos(x) * sin(y);
    }, Kokkos::Sum<double>(sv));
  Kokkos::parallel_reduce("tg_eu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){
      const double x = (i + 0.5) * p.dx;
      const double y = (j + 0.5) * p.dy;
      const double phi = sin(x) * cos(y);
      const double du = u(i,j) - p.U0 * phi;
      acc += du*du;
    }, Kokkos::Sum<double>(eu));
  Kokkos::parallel_reduce("tg_ev", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){
      const double x = (i + 0.5) * p.dx;
      const double y = (j + 0.5) * p.dy;
      const double phi = cos(x) * sin(y);
      const double dv = v(i,j) + p.U0 * phi;
      acc += dv*dv;
    }, Kokkos::Sum<double>(ev));
  TGModeStats s;
  s.Au = su / N;
  s.Av = sv / N;
  s.A = 0.5 * (fabs(s.Au) + fabs(s.Av));
  s.Eu = eu / N;
  s.Ev = ev / N;
  return s;
}

static inline double clamp_sqrt(double x) { return std::sqrt(std::max(0.0, x)); }

static StatsSimple compute_stats_simple(const View2DArray& u, const View2DArray& v, const View2DArray& vort, const Params& p) {
  const double N = double(p.nx) * double(p.ny);
  double sumu=0.0, sumv=0.0, sumu2=0.0, sumv2=0.0, sume=0.0, sumvort2=0.0;
  double maxu=0.0, maxv=0.0, maxspd=0.0;

  Kokkos::parallel_reduce("sumu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ acc += u(i,j); }, Kokkos::Sum<double>(sumu));
  Kokkos::parallel_reduce("sumv", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ acc += v(i,j); }, Kokkos::Sum<double>(sumv));
  Kokkos::parallel_reduce("sumu2", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ const double U=u(i,j); acc += U*U; }, Kokkos::Sum<double>(sumu2));
  Kokkos::parallel_reduce("sumv2", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ const double V=v(i,j); acc += V*V; }, Kokkos::Sum<double>(sumv2));
  Kokkos::parallel_reduce("sume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ const double U=u(i,j), V=v(i,j); acc += 0.5*(U*U + V*V); }, Kokkos::Sum<double>(sume));
  Kokkos::parallel_reduce("sumvort2", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& acc){ const double W=vort(i,j); acc += W*W; }, Kokkos::Sum<double>(sumvort2));

  Kokkos::parallel_reduce("maxu", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& m){ m = fmax(m, fabs(u(i,j))); }, Kokkos::Max<double>(maxu));
  Kokkos::parallel_reduce("maxv", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& m){ m = fmax(m, fabs(v(i,j))); }, Kokkos::Max<double>(maxv));
  Kokkos::parallel_reduce("maxspd", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j, double& m){ const double sp = sqrt(u(i,j)*u(i,j)+v(i,j)*v(i,j)); m = fmax(m, sp); }, Kokkos::Max<double>(maxspd));

  StatsSimple s;
  s.mu = sumu / N; s.mv = sumv / N;
  s.u2 = sumu2 / N; s.v2 = sumv2 / N; s.e = sume / N; s.vort2 = sumvort2 / N;
  s.maxu = maxu; s.maxv = maxv; s.maxspd = maxspd;
  return s;
}

static inline double tg_u0(double x, double y, const Params& p) {
  const double kx = 2.0*M_PI / p.Lx;
  const double ky = 2.0*M_PI / p.Ly;
  return p.U0 * sin(kx*x) * cos(ky*y);
}
static inline double tg_v0(double x, double y, const Params& p) {
  const double kx = 2.0*M_PI / p.Lx;
  const double ky = 2.0*M_PI / p.Ly;
  return -p.U0 * cos(kx*x) * sin(ky*y);
}

static void fill_equilibrium(const D2Q9& lat, const Params& p, const View2DArray& rho, const View2DArray& u, const View2DArray& v, const View3DArray& f) {
  Kokkos::parallel_for("fill_eq", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      const double R = rho(i,j);
      const double U = u(i,j);
      const double V = v(i,j);
      const double uu = U*U + V*V;
      for(int k=0;k<lat.np;++k){
        const double cu = lat.cx[k]*U + lat.cy[k]*V;
        f(i,j,k) = lat.wf[k] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
      }
    });
  Kokkos::fence();
}

static void initialize_tg(View2DArray rho, View2DArray u, View2DArray v, View3DArray f, const D2Q9& lat, const Params& p) {
  Kokkos::parallel_for("init_tg", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      const double x = (i + 0.5) * p.dx;
      const double y = (j + 0.5) * p.dy;
      rho(i,j) = p.rho0;
      u(i,j) = tg_u0(x,y,p);
      v(i,j) = tg_v0(x,y,p);
      const double R = rho(i,j), U = u(i,j), V = v(i,j), uu = U*U + V*V;
      for(int k=0;k<lat.np;++k){
        const double cu = lat.cx[k]*U + lat.cy[k]*V;
        f(i,j,k) = lat.wf[k] * R * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*uu);
      }
    });
  Kokkos::fence();
}

static void compute_vorticity(const View2DArray& u, const View2DArray& v, View2DArray vort, const Params& p) {
  Kokkos::parallel_for("vorticity", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      const int ip = (i + 1) % p.nx, im = (i - 1 + p.nx) % p.nx;
      const int jp = (j + 1) % p.ny, jm = (j - 1 + p.ny) % p.ny;
      const double dvdx = (v(ip,j) - v(im,j)) / (2.0*p.dx);
      const double dudy = (u(i,jp) - u(i,jm)) / (2.0*p.dy);
      vort(i,j) = dvdx - dudy;
    });
  Kokkos::fence();
}

static void write_vtk(const string& prefix, int it, const View2DArray& rho, const View2DArray& u, const View2DArray& v, const View2DArray& vort, const Params& p) {
  View2DArray speed("speed", p.nx, p.ny);
  View2DArray ke("ke", p.nx, p.ny);
  Kokkos::parallel_for("build_aux", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      const double U = u(i,j), V = v(i,j);
      speed(i,j) = sqrt(U*U + V*V);
      ke(i,j) = 0.5*(U*U + V*V);
    });
  Kokkos::fence();

  auto h_rho = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho);
  auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  auto h_vort = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vort);
  auto h_speed = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), speed);
  auto h_ke = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ke);

  const string fname = prefix + "_" + to_string(it) + ".vtk";
  ofstream out(fname);
  if(!out.is_open()) return;

  out << "# vtk DataFile Version 3.0\n";
  out << "TG fluctuating LB output\n";
  out << "ASCII\n";
  out << "DATASET RECTILINEAR_GRID\n";
  out << "DIMENSIONS " << p.nx << " " << p.ny << " 1\n";
  out << "X_COORDINATES " << p.nx << " float\n";
  for(int i=0;i<p.nx;++i) out << (i + 0.5)*p.dx << ' ';
  out << "\nY_COORDINATES " << p.ny << " float\n";
  for(int j=0;j<p.ny;++j) out << (j + 0.5)*p.dy << ' ';
  out << "\nZ_COORDINATES 1 float\n0\n";
  out << "POINT_DATA " << (p.nx*p.ny) << '\n';

  out << "VECTORS Velocity float\n";
  for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i) out << h_u(i,j) << ' ' << h_v(i,j) << " 0\n";
  out << "SCALARS Density float 1\nLOOKUP_TABLE default\n";
  for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i) out << h_rho(i,j) << '\n';
  out << "SCALARS Speed float 1\nLOOKUP_TABLE default\n";
  for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i) out << h_speed(i,j) << '\n';
  out << "SCALARS Vorticity float 1\nLOOKUP_TABLE default\n";
  for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i) out << h_vort(i,j) << '\n';
  out << "SCALARS KineticEnergyDensity float 1\nLOOKUP_TABLE default\n";
  for(int j=0;j<p.ny;++j) for(int i=0;i<p.nx;++i) out << h_ke(i,j) << '\n';
  out.close();
}

static void stream(const D2Q9& lat, const Params& p, const View3DArray& fin, View3DArray fout) {
  Kokkos::parallel_for("stream", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      for(int k=0;k<lat.np;++k){
        const int in = (i + lat.cx[k] + p.nx) % p.nx;
        const int jn = (j + lat.cy[k] + p.ny) % p.ny;
        fout(in, jn, k) = fin(i,j,k);
      }
    });
  Kokkos::fence();
}

static void collide_and_stream(const D2Q9& lat, const Params& p, View3DArray f_in, View3DArray f_out, View2DArray rho, View2DArray u, View2DArray v, RNGPool& rng_pool) {
  Kokkos::parallel_for("collide", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{p.nx,p.ny}),
    KOKKOS_LAMBDA(const int i, const int j){
      double R = 0.0;
      for(int k=0;k<lat.np;++k) R += f_in(i,j,k);
      double U = 0.0, V = 0.0;
      for(int k=0;k<lat.np;++k){ U += lat.cx[k]*f_in(i,j,k); V += lat.cy[k]*f_in(i,j,k); }
      U /= R; V /= R;
      rho(i,j) = R; u(i,j) = U; v(i,j) = V;
      const double U2 = U*U, V2 = V*V, UV = U*V;

      auto rng = rng_pool.get_state();
      const double eta3 = rng.normal();
      const double eta4 = rng.normal();
      const double eta5 = rng.normal();
      const double eta6 = rng.normal();
      const double eta7 = rng.normal();
      const double eta8 = rng.normal();
      rng_pool.free_state(rng);

      double phi3 = std::sqrt(std::max(0.0, p.omega1*(2.0 - p.omega1)*R*p.kBT*lat.b[3]/lat.cs2)) * eta3;
      double phi4 = std::sqrt(std::max(0.0, p.omega*(2.0 - p.omega)*R*p.kBT*lat.b[4]/lat.cs2)) * eta4;
      double phi5 = std::sqrt(std::max(0.0, p.omega*(2.0 - p.omega)*R*p.kBT*lat.b[5]/lat.cs2)) * eta5;
      double phi6 = std::sqrt(std::max(0.0, p.omega1*(2.0 - p.omega1)*R*p.kBT*lat.b[6]/lat.cs2)) * eta6;
      double phi7 = std::sqrt(std::max(0.0, p.omega1*(2.0 - p.omega1)*R*p.kBT*lat.b[7]/lat.cs2)) * eta7;
      double phi8 = std::sqrt(std::max(0.0, p.omega1*(2.0 - p.omega1)*R*p.kBT*lat.b[8]/lat.cs2)) * eta8;

      double r0 = R;
      double r1 = R*U;
      double r2 = R*V;
      double r3 = R*(U2 + V2) + phi3;
      double r4 = R*(U2 - V2) + phi4;
      double r5 = R*UV + phi5;
      double r6 = R*U2*V + phi6;
      double r7 = R*U*V2 + phi7;
      double r8 = R*U2*V2 + phi8;

      const double f0 = 4.*r0/9. - 2.*r3/3. + r8;
      const double f1 = r4/4. - (r7 + r8)/2. + r3/12. + r0/9. + r1/3.;
      const double f2 = r3/12. - r4/4. - (r6 + r8)/2. + r0/9. + r2/3.;
      const double f3 = r7/2. - r8/2. + r4/4. + r3/12. + r0/9. - r1/3.;
      const double f4 = r6/2. - r8/2. - r4/4. + r3/12. + r0/9. - r2/3.;
      const double f5 = (r5 + r6 + r7 + r8)/4. + (r1 + r2 + r3)/12. + r0/36.;
      const double f6 = (r6 + r8)/4. - (r5 + r7)/4. + (r2 + r3)/12. - r1/12. + r0/36.;
      const double f7 = (r5 + r8)/4. - (r6 + r7)/4. + r3/12. - (r1 + r2)/12. + r0/36.;
      const double f8 = (r7 + r8)/4. - (r5 + r6)/4. + (r1 + r3)/12. - r2/12. + r0/36.;

      f_in(i,j,0) = f0; f_in(i,j,1) = f1; f_in(i,j,2) = f2; f_in(i,j,3) = f3; f_in(i,j,4) = f4;
      f_in(i,j,5) = f5; f_in(i,j,6) = f6; f_in(i,j,7) = f7; f_in(i,j,8) = f8;
    });
  Kokkos::fence();
  stream(lat, p, f_in, f_out);
}


static void write_tg_csv_header(const string& fname) {
  ofstream out(fname);
  out << "it,t,Au,Av,A,Eu,Ev,mu,mv,u2,v2,E,vort2,maxspd";
}

static void append_tg_csv(const string& fname, int it, double t, const TGModeStats& tg, const StatsSimple& s) {
  ofstream out(fname, ios::app);
  out << it << ',' << t << ',' << tg.Au << ',' << tg.Av << ',' << tg.A << ',' << tg.Eu << ',' << tg.Ev << ','
      << s.mu << ',' << s.mv << ',' << s.u2 << ',' << s.v2 << ',' << s.e << ',' << s.vort2 << ',' << s.maxspd << '\n';
}

static void run_case_tg_fluct() {
  cout << "\n=== 2D decaying Taylor-Green vortex with fluctuations ===\n";
  Params p;
  p.nx = 128;
  p.ny = 128;
  p.U0 = 0.01;
  p.ni = 0.0001;
  p.kBT = 1.0/3000.0;
  p.nsteps = 4000;
  p.n_out = 100;
  p.finalize();

  D2Q9 lat;
  View3DArray f1("f1", p.nx, p.ny, lat.np);
  View3DArray f2("f2", p.nx, p.ny, lat.np);
  View2DArray rho("rho", p.nx, p.ny);
  View2DArray u("u", p.nx, p.ny);
  View2DArray v("v", p.nx, p.ny);
  View2DArray vort("vort", p.nx, p.ny);
  RNGPool rng_pool(123456789ULL);

  initialize_tg(rho, u, v, f1, lat, p);
  stream(lat, p, f1, f2);
  std::swap(f1, f2);

  const string csv = "TG2D_fluct.csv";
  write_tg_csv_header(csv);

  for(int it=0; it<=p.nsteps; ++it) {
    collide_and_stream(lat, p, f1, f2, rho, u, v, rng_pool);
    std::swap(f1, f2);

    if(it % p.n_out == 0) {
      compute_vorticity(u, v, vort, p);
      auto s = compute_stats_simple(u, v, vort, p);
      auto tg = compute_tg_mode(u, v, p);
      cout << fixed << setprecision(8)
           << "it=" << it
           << " t=" << (it * p.dx / p.U0)
           << " Au=" << tg.Au
           << " Av=" << tg.Av
           << " A=" << tg.A
           << " <u>=" << s.mu
           << " <v>=" << s.mv
           << " <u2>=" << s.u2
           << " <v2>=" << s.v2
           << " E=" << s.e
           << " max|u|=" << s.maxspd
           << " <w2>=" << s.vort2
           << endl;
      append_tg_csv(csv, it, it * p.dx / p.U0, tg, s);
      write_vtk("TG2D_fluct", it, rho, u, v, vort, p);
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    run_case_tg_fluct();
  }
  Kokkos::finalize();
  return 0;
}
