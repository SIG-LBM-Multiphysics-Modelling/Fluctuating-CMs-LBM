reset
set terminal epslatex standalone color colortext size 9cm,6cm
set output 'vel_pdf_hist.tex'

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

# --- binning (choose bin width so that 0 is a bin centre)
xmin = -5.0
xmax =  5.0
nbins = 30
binw  = (xmax-xmin)/nbins

bin(x) = binw*floor((x-xmin)/binw) + 0.5*binw

# total samples
N = SX_records
norm = 1.0/(N*binw)

pi = 3.141592653589793
gauss(x) = 1.0/sqrt(2.0*pi)*exp(-0.5*x*x)

set style fill solid 0.35 border -1
set boxwidth binw absolute

plot \
  file using (bin((column(colx)-mux)/sigx)):(norm) smooth freq w boxes lc rgb '#4B0082' title '$u_x$', \
  ''   using (bin((column(coly)-muy)/sigy)):(norm) smooth freq w boxes lc rgb '#008B8B' title '$u_y$', \
  ''   using (bin((column(colz)-muz)/sigz)):(norm) smooth freq w boxes lc rgb '#BDB76B' title '$u_z$', \
  gauss(x) w l lw 2 dt 2 lc 'black' title 'Gaussian'
