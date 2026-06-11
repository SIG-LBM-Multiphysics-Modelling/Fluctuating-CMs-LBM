reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test6.tex'
set yr[-0.2:6]
set xr[0.42:115]
set ytics 1
set xtics ('0.5' 0.5, '1' 1, '5' 5, '10' 10, '50' 50, '100' 100) 
set key top right Right samplen 1 
set xlabel '$\tau$' offset 0,0.7
set ylabel '$\psi$'  offset 0.4,0
set logscale x
set pointsize 2
# Reference  
g(x) = 0
plot  "test6BGK.txt" u 1:(abs($4-$5)/$5*100) w lp lc 'green' dt 1 lw 3 pt 12 title 'BGK',\
     "test6MRT.txt" u 1:(abs($4-$5)/$5*100) w lp lc 'blue' dt 1 lw 3 pt 14 title 'MRT',\
     "test6.txt" u 1:(abs($4-$5)/$5*100) w lp lc 'red' dt 1 lw 3 pt 16 title 'CMs' 


reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test6_zoom.tex'
#set yr[0:6]
set xr[0.4995:0.5105]
#set ytics 0.5
set xtics ('0.5' 0.5, '0.502' 0.502, '0.504' 0.504, '0.506' 0.506, '0.508' 0.508, '0.51' 0.51) 
set key top Left left samplen 1 
unset key
set xlabel '$\tau$' offset 0,0.7
set ylabel '$\psi$'  offset 0.5,0
set logscale x
set pointsize 2
# Reference  
g(x) = 0
plot  "test6BGK.txt" u 1:(abs($4-$5)/$5*100) w p lc 'green' dt 1 lw 3 pt 12 title 'BGK',\
     "test6MRT.txt" u 1:(abs($4-$5)/$5*100) w p lc 'blue' dt 1 lw 3 pt 14 title 'MRT',\
     "test6.txt" u 1:(abs($4-$5)/$5*100) w p lc 'red' dt 1 lw 3 pt 16 title 'CMs' 


reset
set colorsequence default
set terminal epslatex standalone color colortext size 9cm,6cm
set out 'Test6_zoom2.tex'
#set yr[0:6]
#set ytics 1
set xr[0.4985:0.5515]
set xtics ('0.5' 0.5, '0.51' 0.51, '0.52' 0.52, '0.53' 0.53, '0.54' 0.54, '0.55' 0.55) 
#set logscale x
set key top right Right samplen 1 
set xlabel '$\tau$' offset 0,0.7
set ylabel '$\psi$'  offset 0.5,0
set pointsize 2
plot  "test6BGK.txt" u 1:(abs($4-$5)/$5*100) w p lc 'green' dt 1 lw 3 pt 12 title 'BGK',\
     "test6MRT.txt" u 1:(abs($4-$5)/$5*100) w p lc 'blue' dt 1 lw 3 pt 14 title 'MRT',\
     "test6.txt" u 1:(abs($4-$5)/$5*100) w p lc 'red' dt 1 lw 3 pt 16 title 'CMs' 

