reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test4.tex'
set yr[0.00015:0.0015]
set xr[0.00015:0.0015]
set ytics ('1/6000' 0.0001666666667, '1/3000' 0.0003333333333, '1/1500' 0.0006666666667, '1/750' 0.001333333333)
set xtics ('1/6000' 0.0001666666667, '1/3000' 0.0003333333333, '1/1500' 0.0006666666667, '1/750' 0.001333333333)
set key bottom right Right samplen 1 spacing 1.4
set xlabel '$k_B \,T/\rho_0$' offset 0,0.2
set ylabel '$\langle u_\alpha^2\rangle$' offset 1,0
set logscale x
set logscale y
set pointsize 2
plot  "test4.txt" u 1:2 w lp lc 'dark-orange' dt 1 lw 3 pt 4 title '$\langle u_x^2\rangle$' ,\
	  "test4.txt" u 1:3 w lp lc 'dark-yellow' dt 1 lw 3 pt 6 title '$\langle u_y^2\rangle$' ,\
	  "test4.txt" u 1:1 w l lc -1 dt 2 lw 3 title 'Expected' 



