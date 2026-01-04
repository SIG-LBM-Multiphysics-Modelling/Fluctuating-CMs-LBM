reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test7-Case1.tex'
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
plot "case1.txt" u 1:(($2-$5)/$5*100) w lp lc 'dark-violet' dt 1 pt 5 lw 3 title '$u_x$' ,\
"case1.txt" u 1:(($3-$5)/$5*100) w lp lc 'dark-turquoise' dt 1 pt 7 lw 3 title '$u_y$' ,\
"case1.txt" u 1:(($4-$5)/$5*100) w lp lc 'dark-khaki' dt 1 pt 9 lw 3 title '$u_y$' ,\
 g(x) w l lw 3 dt 2 lc 'black' title 'Expected'

 reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test7-Case2.tex'
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
plot "case2.txt" u 1:(($2-$5)/$5*100) w lp lc 'dark-violet' dt 1 pt 5 lw 3 title '$u_x$' ,\
"case2.txt" u 1:(($3-$5)/$5*100) w lp lc 'dark-turquoise' dt 1 pt 7 lw 3 title '$u_y$' ,\
"case2.txt" u 1:(($4-$5)/$5*100) w lp lc 'dark-khaki' dt 1 pt 9 lw 3 title '$u_y$' ,\
 g(x) w l lw 3 dt 2 lc 'black' title 'Expected'


reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test7-Case3.tex'
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
plot "case3.txt" u 1:(($2-$5)/$5*100) w lp lc 'dark-violet' dt 1 pt 5 lw 3 title '$u_x$' ,\
"case3.txt" u 1:(($3-$5)/$5*100) w lp lc 'dark-turquoise' dt 1 pt 7 lw 3 title '$u_y$' ,\
"case3.txt" u 1:(($4-$5)/$5*100) w lp lc 'dark-khaki' dt 1 pt 9 lw 3 title '$u_y$' ,\
 g(x) w l lw 3 dt 2 lc 'black' title 'Expected'


 reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test7-Case4.tex'
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
plot "case4.txt" u 1:(($2-$5)/$5*100) w lp lc 'dark-violet' dt 1 pt 5 lw 3 title '$u_x$' ,\
"case4.txt" u 1:(($3-$5)/$5*100) w lp lc 'dark-turquoise' dt 1 pt 7 lw 3 title '$u_y$' ,\
"case4.txt" u 1:(($4-$5)/$5*100) w lp lc 'dark-khaki' dt 1 pt 9 lw 3 title '$u_y$' ,\
 g(x) w l lw 3 dt 2 lc 'black' title 'Expected'


