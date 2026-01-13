reset
set terminal epslatex standalone color colortext size 9cm,6cm
set output 'vel_pdf.tex'

set key top right
set xlabel '$u_\alpha/\sigma_\alpha$'
set ylabel '$\mathcal{P}(u_\alpha/\sigma_\alpha)$'
set grid
set xrange [-5:5]
set yrange [0:*]

file = 'vel.txt'
colx=1; coly=2; colz=3

# stats
stats file using colx name 'SX' nooutput
stats file using coly name 'SY' nooutput
stats file using colz name 'SZ' nooutput

mux=SX_mean; sigx=SX_stddev
muy=SY_mean; sigy=SY_stddev
muz=SZ_mean; sigz=SZ_stddev

pi=3.141592653589793
gauss(x)=1.0/sqrt(2.0*pi)*exp(-0.5*x*x)

# bandwidth in normalised units (try 0.15–0.25)
h = 0.20

plot \
  file using ((column(colx)-mux)/sigx):(1.0) w l lw 3 title '$u_x$', \
  ''   using ((column(coly)-muy)/sigy):(1.0) w l lw 3 title '$u_y$', \
  ''   using ((column(colz)-muz)/sigz):(1.0) w l lw 3 title '$u_z$', \
  gauss(x) w l lw 2 dt 2 title 'Gaussian'
