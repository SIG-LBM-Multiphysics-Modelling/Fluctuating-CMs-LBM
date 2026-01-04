reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test6.tex'
set yr[-0.05:2.5]
set xr[0.45:120]
set ytics 0.5
set xtics ('0.5' 0.5, '1' 1, '5' 5, '10' 10, '50' 50, '100' 100) 
set key top right Right samplen 1 
set xlabel '$\tau$' offset 0,0.7
set ylabel '$\psi$'  offset 1.5,0
set logscale x
set pointsize 2
# Reference  
g(x) = 0
plot  "test6BGK.txt" u 1:(abs($4-$5)/$5*100) w lp lc 'green' dt 1 lw 3 pt 12 title 'BGK',\
     "test6.txt" u 1:(abs($4-$5)/$5*100) w lp lc 'red' dt 1 lw 3 pt 14 title 'Present' 


reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test6_zoom.tex'
set yr[0.9:2.5]
set xr[0.4995:0.5105]
set ytics 0.5
#set xtics ('0.5' 0.5, '1' 1, '5' 5, '10' 10, '50' 50, '100' 100) 
set key top Left left samplen 1 
unset key
set xlabel '$\tau$' offset 0,0.7
set ylabel '$\psi$'  offset 1.5,0
set logscale x
set pointsize 2
# Reference  
g(x) = 0
plot  "test6BGK.txt" u 1:(abs($4-$5)/$5*100) w p lc 'green' dt 1 lw 3 pt 12 title 'BGK',\
     "test6.txt" u 1:(abs($4-$5)/$5*100) w p lc 'red' dt 1 lw 3 pt 14 title 'Present' 



