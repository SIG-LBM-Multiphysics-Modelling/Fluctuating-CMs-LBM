reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out '3DTG.tex'

set yr[0.01:2]
set xr[0:10]
set key top right Right samplen 1 
set xlabel '$t/t_0$' offset 0,0.7
set ylabel '$E \times 1000$' offset 0.8,0
set logscale y

plot  "energy_enstrophy.txt" u 2:($3*1000) w l lc 'red' dt 1 lw 3 notitle smooth bezier, \
      0.015 w l lc 'black' dt 2 lw 2 title '$E_{k_B T} \times 1000$'