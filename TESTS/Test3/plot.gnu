reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test3.tex'
set yr[-3:3]
set xr[-1:102]
set xtics 20
set ytics 1
set key top right Right samplen 1
unset key
set xlabel '$t/t_0$' offset 0,0.7
set ylabel '$\theta_{\alpha}$'  offset 0.5,0
set pointsize 2
# Reference 
g(x) = 0
plot "equipartition.txt" u 1:(($2-$4)/$4*100) w lp lc 'dark-magenta' dt 1 pt 8 lw 3 title '$u_x$' ,\
"equipartition.txt" u 1:(($3-$4)/$4*100) w lp lc 'dark-cyan' dt 1 pt 10 lw 3 title '$u_y$' ,\
 g(x) w l lw 3 dt 2 lc 'black' title 'Expected'



