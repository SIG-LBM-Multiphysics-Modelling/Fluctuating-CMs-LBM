reset
set colorsequence classic
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test5.tex'
set yr[0.00008:0.00071]
set xr[0.47:4.2]
set ytics ('$1/12000$' 0.0000833333333, '$1/6000$'  0.0001666666667, '$1/3000$'  0.0003333333333, '$1/1500$'  0.0006666666667)
set xtics ('0.5' 0.5, '1' 1, '2' 2, '4' 4)
set key top right Right samplen 1 spacing 1.4
set xlabel '$\rho_0$' offset 0,1
set ylabel '$\langle u_\alpha^2\rangle$' offset 2,0
set logscale x
set logscale y
set pointsize 2
plot  "test5.txt" u 1:2 w lp lc 'goldenrod' dt 1 lw 3 pt 9 title '$\langle u_x^2\rangle$' ,\
	  "test5.txt" u 1:3 w lp lc 'dark-spring-green' dt 1 lw 3 pt 11 title '$\langle u_y^2\rangle$' ,\
	  "test5.txt" u 1:5 w l lc  -1 dt 2 lw 3 title 'Expected' 



