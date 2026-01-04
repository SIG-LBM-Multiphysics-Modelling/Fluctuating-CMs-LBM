reset
set colorsequence classic
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test2.tex'
set yr[-5:0]
set xr[0.5:3]
set xtics 0.5
set ytics 1
unset key
set pointsize 2
set label 10 '$\mathrm{slope}=-2$' at 1,-3 front
# Reference slope -2 line through (1,-2) and (1.5,-3)
g(x) = -2*x+0.6
set xlabel '$\mathrm{log}_{10}(N_y)$' offset 0,0.5
set ylabel '$\mathrm{log}_{10}(\varepsilon)$' 

plot "error.txt" u (log10($1)):(log10($2)) w lp lc 'blue' dt 2 pt 3 lw 3 ,\
 (x>=1.2 && x<=2 ? g(x) : 1/0) w l lw 2 lc 'black'


