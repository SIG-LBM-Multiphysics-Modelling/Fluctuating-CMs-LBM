reset
set term epslatex standalone color colortext size 9cm,6cm
set out 'Test5_relerr_inset.tex'
set colorsequence classic

PRESENT = "test5.txt"
BGK     = "test5BGK.txt"

# Relative error (%)
relerr(m,t) = abs(m - t)/t * 100.0

# --- find max finite error (for NaN marker placement)
stats PRESENT using (relerr($4,$5)) nooutput
maxP = STATS_max
stats BGK using (relerr($4,$5)) nooutput
maxB = STATS_max
maxE = (maxP > maxB ? maxP : maxB)

yNaN = 1.2*maxE
yTop = 1.4*maxE

set multiplot

# ================= MAIN PLOT =================
set logscale x
set xrange [0.47:120]
set yrange [0:yTop]

set xlabel '$\tau$'
set ylabel 'relative error (\%)'
set key top right
set pointsize 1.2

plot \
  PRESENT using 1:(relerr($4,$5)) \
      with lp pt 7 lw 2 title 'Present', \
  BGK using 1:( ($4==$4) ? relerr($4,$5) : 1/0 ) \
      with lp pt 5 lw 2 title 'BGK', \
  BGK using 1:( ($4!=$4) ? yNaN : 1/0 ) \
      with points pt 2 ps 1.4 title 'BGK (NaN)'

# ================= INSET =================
set size 0.42,0.42
set origin 0.52,0.18

unset key
unset logscale x
set xrange [0.4995:0.512]
set yrange [0:yTop]

set xlabel ''
set ylabel ''
set format x '%.4f'
set tics scale 0.7

plot \
  PRESENT using 1:(relerr($4,$5)) \
      with lp pt 7 lw 1 notitle, \
  BGK using 1:( ($4==$4) ? relerr($4,$5) : 1/0 ) \
      with lp pt 5 lw 1 notitle, \
  BGK using 1:( ($4!=$4) ? yNaN : 1/0 ) \
      with points pt 2 ps 1.2 notitle

unset multiplot

