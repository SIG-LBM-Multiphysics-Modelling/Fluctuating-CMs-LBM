%--------------------------------------------------------------------------
% This program is part of the paper "?Multiphysics flow simulations at low
% cost using D3Q19 lattice Boltzmann methods based on central moments".
% Copyright (C) 2020  Alessandro De Rosis (derosis.alessandro@icloud.com)
% 
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% This is the "D3Q19_CentralMoments.m" file.
% Author: Alessandro De Rosis (derosis.alessandro@icloud.com)
% Day: 1st Jun 2020
%--------------------------------------------------------------------------

clear all
clc
syms U V W omega k5_star k6_star k7_star k8_star k9_star Fx Fy Fz kBT R...
     eta4 eta5 eta6 eta7 eta8 eta9 eta10 eta11 eta12 eta13 eta14 eta15 eta16 eta17 eta18 real

%% Define useful quantities
central_moments = true;
forcing = false;
noise = true;
% Relaxation matrix
L = diag([1 1 1 1 1 1 1 omega omega omega 1 1 1 1 1 1 1 1 1]);
cx = [ 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 ];
cy = [ 0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 ];
cz = [ 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 ];
eta = [0 0 0 0 eta4 eta5 eta6 eta7 eta8 eta9 eta10 eta11 eta12 eta13 eta14 eta15 eta16 eta17 eta18]';
w  = [ 1/3, ...
       1/18,1/18,1/18,1/18,1/18,1/18, ...
       1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36 ];
cs = 1/sqrt(3);
cs = 1/sqrt(3);
cs2 = cs^2;
cs3 = cs^3;
cs4 = cs^4;
cs5 = cs^5;
cs6 = cs^6;
cs8 = cs^8;
cs10 = cs^10;
cs12 = cs^12;
T = sym(zeros(19,19));
M = zeros(19,19);
N = sym(zeros(19,19));
feq = sym(zeros(19,1));
geq = sym(zeros(19,1));
force_guo = sym(zeros(19,1));
Force = sym(zeros(19,1));
G = zeros(19,19);

U2 = U*U;
V2 = V*V;
W2 = W*W;
u2 = U*U+V*V+W*W;
u = [U, V, W];
for i=1:length(cx)
    % build equilibrium populations
    if(cx(i)==0 && cy(i)==0 && cz(i)==0)
        feq(i) = R/3*(1-(U2+V2+W2)+3*(U2*V2+U2*W2+V2*W2));
    elseif(cx(i)~=0 && cy(i)==0 && cz(i)==0)
        feq(i) = R/18*(1+3*cx(i)*U+3*(U2-V2-W2)-9*cx(i)*(U*V2+U*W2)-9*(U2*V2+U2*W2));
    elseif(cx(i)==0 && cy(i)~=0 && cz(i)==0)
        feq(i) = R/18*(1+3*cy(i)*V+3*(-U2+V2-W2)-9*cy(i)*(U2*V+V*W2)-9*(U2*V2+V2*W2));
    elseif(cx(i)==0 && cy(i)==0 && cz(i)~=0)
        feq(i) = R/18*(1+3*cz(i)*W+3*(-U2-V2+W2)-9*cz(i)*(U2*W+V2*W)-9*(U2*W2+V2*W2));
    elseif(cx(i)~=0 && cy(i)~=0 && cz(i)==0)
        feq(i) = R/36*(1+3*(cx(i)*U+cy(i)*V)+3*(U2+V2)+9*cx(i)*cy(i)*U*V+...
                 9*(cy(i)*U2*V+cx(i)*U*V2)+9*U2*V2);
    elseif(cx(i)~=0 && cy(i)==0 && cz(i)~=0)
        feq(i) = R/36*(1+3*(cx(i)*U+cz(i)*W)+3*(U2+W2)+9*cx(i)*cz(i)*U*W+...
                 9*(cz(i)*U2*W+cx(i)*U*W2)+9*U2*W2);
    elseif(cx(i)==0 && cy(i)~=0 && cz(i)~=0)
        feq(i) = R/36*(1+3*(cy(i)*V+cz(i)*W)+3*(V2+W2)+9*cy(i)*cz(i)*V*W+...
                 9*(cz(i)*V2*W+cy(i)*V*W2)+9*V2*W2);
    end
    
  cxi = cx(i);  cyi = cy(i);  czi = cz(i);
    cx2 = cxi*cxi;  cy2 = cyi*cyi;  cz2 = czi*czi;

    % number of non-zero components (0: rest, 1: axis, 2: face diagonal)
    nnz = cx2 + cy2 + cz2;

    if nnz == 0
        % ---- rest (0,0,0) ----
        feq(i) = R*( ...
              U^4/2 + U^2*V^2 + U^2*W^2 + U^2/6 ...
            + V^4/2 + V^2*W^2 + V^2/6 ...
            + W^4/2 + W^2/6 ...
            + 1/3 );

    elseif nnz == 1
        % ---- axis (±1,0,0) or (0,±1,0) or (0,0,±1) ----
        if cx2 == 1
            s = cxi; % s = ±1
            feq(i) = -R*( ...
                -U^4/2 + (s)*U^3/2 + (U^2*V^2)/2 + (U^2*W^2)/2 + U^2/3 ...
                + (s)*(U*V^2)/2 + (s)*(U*W^2)/2 - (s)*U/6 ...
                + V^4/2 + V^2/6 + W^4/2 + W^2/6 - 1/18 );

        elseif cy2 == 1
            s = cyi; % s = ±1
            feq(i) = -R*( ...
                U^4/2 + (U^2*V^2)/2 + (s)*(U^2*V)/2 + U^2/6 ...
              - V^4/2 + (s)*V^3/2 + (V^2*W^2)/2 + V^2/3 ...
              + (s)*(V*W^2)/2 - (s)*V/6 ...
              + W^4/2 + W^2/6 - 1/18 );

        else % cz2 == 1
            s = czi; % s = ±1
            feq(i) = -R*( ...
                U^4/2 + (U^2*W^2)/2 + (s)*(U^2*W)/2 + U^2/6 ...
              + V^4/2 + (V^2*W^2)/2 + (s)*(V^2*W)/2 + V^2/6 ...
              - W^4/2 + (s)*W^3/2 + W^2/3 - (s)*W/6 - 1/18 );
        end

    else
        % ---- face diagonals (two nonzero components, one zero) ----
        if cz2 == 0
            % (±1,±1,0)  -> signs s1=cx, s2=cy, W is the "zero" component
            s1 = cxi;  s2 = cyi;
            feq(i) = R*( ...
                -U^4/8 + (s1)*U^3/8 + (U^2*V^2)/4 + (s2)*(U^2*V)/4 + (5*U^2)/24 ...
                + (s1)*(U*V^2)/4 + (s1*s2)*(U*V)/4 + (s1)*U/12 ...
                -V^4/8 + (s2)*V^3/8 + (5*V^2)/24 + (s2)*V/12 ...
                + (3*W^4)/8 - W^2/8 + 1/36 );

        elseif cy2 == 0
            % (±1,0,±1)  -> signs s1=cx, s3=cz, V is the "zero" component
            s1 = cxi;  s3 = czi;
            feq(i) = R*( ...
                -U^4/8 + (s1)*U^3/8 + (U^2*W^2)/4 + (s3)*(U^2*W)/4 + (5*U^2)/24 ...
                + (s1)*(U*W^2)/4 + (s1*s3)*(U*W)/4 + (s1)*U/12 ...
                + (3*V^4)/8 - V^2/8 ...
                -W^4/8 + (s3)*W^3/8 + (5*W^2)/24 + (s3)*W/12 ...
                + 1/36 );

        else
            % (0,±1,±1)  -> signs s2=cy, s3=cz, U is the "zero" component
            s2 = cyi;  s3 = czi;
            feq(i) = R*( ...
                + (3*U^4)/8 - U^2/8 ...
                -V^4/8 + (s2)*V^3/8 + (V^2*W^2)/4 + (s3)*(V^2*W)/4 + (5*V^2)/24 ...
                + (s2)*(V*W^2)/4 + (s2*s3)*(V*W)/4 + (s2)*V/12 ...
                -W^4/8 + (s3)*W^3/8 + (5*W^2)/24 + (s3)*W/12 ...
                + 1/36 );
        end
    end

%     % Set the basis
%     CX = cx(i)-U;
%     CY = cy(i)-V;
%     CZ = cz(i)-W;
%     CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
%     C2  = CX2 + CY2 + CZ2;
%     T(1,i)  = 1;
%     T(2,i)  = CX;
%     T(3,i)  = CY;
%     T(4,i)  = CZ;
%     T(5,i)  = C2-1;
%     T(6,i)  = 2*CX2-CY2-CZ2;
%     T(7,i)  = CY2-CZ2;
%     T(8,i)  = CX*CY;
%     T(9,i)  = CX*CZ;
%     T(10,i) = CY*CZ;
%     T(11,i) = CX*(C2-5/3);
%     T(12,i) = CY*(C2-5/3);
%     T(13,i) = CZ*(C2-5/3); 
%     T(14,i) = CX*(CY2-CZ2);
%     T(15,i) = CY*(CZ2-CX2);
%     T(16,i) = CZ*(CX2-CY2);
%     T(17,i) = (C2-1)^2 - 2/3;
%     T(18,i) = (CY2 - CZ2)*(CX2 - 1/2);
%     T(19,i) = (CX2*CY2 + CX2*CZ2 - 2*CY2*CZ2) - 0.5*(2*CX2 - CY2 - CZ2);
%     
%     CX  = cx(i);  CY  = cy(i);  CZ  = cz(i);
%     CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
%     C2  = CX2 + CY2 + CZ2;
% 
%     M(1,i)  = 1;
%     M(2,i)  = CX;
%     M(3,i)  = CY;
%     M(4,i)  = CZ;
%     M(5,i)  = C2-1;
%     M(6,i)  = 2*CX2-CY2-CZ2;
%     M(7,i)  = CY2-CZ2;
%     M(8,i)  = CX*CY;
%     M(9,i)  = CX*CZ;
%     M(10,i) = CY*CZ;
%     M(11,i) = CX*(C2-5/3);
%     M(12,i) = CY*(C2-5/3);
%     M(13,i) = CZ*(C2-5/3); 
%     M(14,i) = CX*(CY2-CZ2);
%     M(15,i) = CY*(CZ2-CX2);
%     M(16,i) = CZ*(CX2-CY2);
%     M(17,i) = (C2 - 1)^2 - 2/3;
%     M(18,i) = (CY2 - CZ2)*(CX2 - 1/2);
%     M(19,i) = (CX2*CY2 + CX2*CZ2 - 2*CY2*CZ2) - 0.5*(2*CX2 - CY2 - CZ2);

% Set the basis
    CX = cx(i)-U;
    CY = cy(i)-V;
    CZ = cz(i)-W;
    CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
    C2  = CX2 + CY2 + CZ2;
    T(1,i)  = 1;
    T(2,i)  = CX;
    T(3,i)  = CY;
    T(4,i)  = CZ;
    T(5,i)  = C2-1;
    T(6,i)  = 3*CX2-C2;
    T(7,i)  = CY2-CZ2;
    T(8,i)  = CX*CY;
    T(9,i)  = CY*CZ;
    T(10,i) = CX*CZ;
    T(11,i) = CX*(3*C2-5);
    T(12,i) = CY*(3*C2-5);
    T(13,i) = CZ*(3*C2-5); 
    T(14,i) = CX*(CY2-CZ2);
    T(15,i) = CY*(CZ2-CX2);
    T(16,i) = CZ*(CX2-CY2);
    T(17,i) = 3*C2*C2 - 6*C2+1;
    T(18,i) = (2*C2-3)*(3*CX2-C2);
    T(19,i) = (2*C2-3)*(CY2-CZ2);
    
    CX  = cx(i);  CY  = cy(i);  CZ  = cz(i);
    CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
    C2  = CX2 + CY2 + CZ2;

    M(1,i)  = 1;
    M(2,i)  = CX;
    M(3,i)  = CY;
    M(4,i)  = CZ;
    M(5,i)  = C2-1;
    M(6,i)  = 3*CX2-C2;
    M(7,i)  = CY2-CZ2;
    M(8,i)  = CX*CY;
    M(9,i)  = CY*CZ;
    M(10,i) = CX*CZ;
    M(11,i) = CX*(3*C2-5);
    M(12,i) = CY*(3*C2-5);
    M(13,i) = CZ*(3*C2-5); 
    M(14,i) = CX*(CY2-CZ2);
    M(15,i) = CY*(CZ2-CX2);
    M(16,i) = CZ*(CX2-CY2);
    M(17,i) = 3*C2*C2 - 6*C2+1;
    M(18,i) = (2*C2-3)*(3*CX2-C2);
    M(19,i) = (2*C2-3)*(CY2-CZ2);

    A = U*cx(i)+V*cy(i)+W*cz(i);
    force_guo(i) = w(i)*( Fx*(3*(cx(i)-U)+9*A*cx(i))+...
                          Fy*(3*(cy(i)-V)+9*A*cy(i))+...
						  Fz*(3*(cz(i)-W)+9*A*cz(i)) );
end
b = sym(zeros(length(cx),length(cx)));
phi = sym(zeros(length(cx),1));
Xi = sym(zeros(length(cx),1));
for k=1:length(cx)
    for l=1:length(cx)
        for i=1:length(cx)
            b(k,l) = b(k,l) + w(i)*M(k,i)*M(l,i);
        end
    end
    phi(k) = sqrt(R*kBT*L(k,k)*(2-L(k,k))*b(k,k)/cs2);
    Xi(k) = phi(k)*eta(k);
end
b
if(central_moments)
    T = simplify(T);
    %N = simplify(T*M^(-1)); %shift matrix
    % N = [1,        0,        0,        0,             0,                 0,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     -U,        1,        0,        0,             0,                 0,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     -V,        0,        1,        0,             0,                 0,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     -W,        0,        0,        1,             0,                 0,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     U^2 + V^2 + W^2,     -2*U,     -2*V,     -2*W,             1,                 0,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     U^2 - V^2,     -2*U,      2*V,        0,             0,                 1,                 0,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     V^2 - W^2,        0,     -2*V,      2*W,             0,                 0,                 1,     0,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     U*V,       -V,       -U,        0,     ,         0,                 0,                 0,     1,     0,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     U*W,       -W,        0,       -U,             0,                 0,                 0,     0,     1,     0,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     V*W,        0,       -W,       -V,             0,                 0,                 0,     0,     0,     1,    0,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     -U^2*V,    2*U*V,      U^2,        0,          -V/3,          -(2*V)/3,              -V/3,  -2*U,     0,     0,    1,    0,    0,    0,    0,    0, 0, 0, 0;...
    %     -U*V^2,      V^2,    2*U*V,        0,          -U/3,               U/3,              -U/3,  -2*V,     0,     0,    0,    1,    0,    0,    0,    0, 0, 0, 0;...
    %     -U^2*W,    2*U*W,        0,      U^2,          -W/3,          -(2*W)/3,              -W/3,     0,  -2*U,     0,    0,    0,    1,    0,    0,    0, 0, 0, 0;...
    %     -U*W^2,      W^2,        0,    2*U*W,          -U/3,               U/3,           (2*U)/3,     0,  -2*W,     0,    0,    0,    0,    1,    0,    0, 0, 0, 0;...
    %     -V^2*W,        0,    2*V*W,      V^2,          -W/3,               W/3,              -W/3,     0,     0,  -2*V,    0,    0,    0,    0,    1,    0, 0, 0, 0;...
    %     -V*W^2,        0,      W^2,    2*V*W,          -V/3,               V/3,           (2*V)/3,     0,     0,  -2*W,    0,    0,    0,    0,    0,    1, 0, 0, 0;...
    %     U^2*V^2, -2*U*V^2, -2*U^2*V,        0, U^2/3 + V^2/3, (2*V^2)/3 - U^2/3,     U^2/3 + V^2/3, 4*U*V,     0,     0, -2*V, -2*U,    0,    0,    0,    0, 1, 0, 0;...
    %     U^2*W^2, -2*U*W^2,        0, -2*U^2*W, U^2/3 + W^2/3, (2*W^2)/3 - U^2/3, W^2/3 - (2*U^2)/3,     0, 4*U*W,     0,    0,    0, -2*W, -2*U,    0,    0, 0, 1, 0;...
    %     V^2*W^2,        0, -2*V*W^2, -2*V^2*W, V^2/3 + W^2/3,   - V^2/3 - W^2/3, W^2/3 - (2*V^2)/3,     0,     0, 4*V*W,    0,    0,    0,    0, -2*W, -2*V, 0, 0, 1];
    % N = simplify(N);
    N(1,:)  = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    N(2,1) = -U;      N(2,2)=1;
    N(3,1) = -V;      N(3,3)=1;
    N(4,1) = -W;      N(4,4)=1;
    N(5,1) = U^2+V^2+W^2;
    N(5,2) = -2*U; N(5,3)=-2*V; N(5,4)=-2*W; N(5,5)=1;

    N(6,1) = U^2 - V^2;
    N(6,2) = -2*U; N(6,3)= 2*V; N(6,6)=1;

    N(7,1) = V^2 - W^2;
    N(7,3) = -2*V; N(7,4)= 2*W; N(7,7)=1;

    N(8,1) = U * V;
    N(8,2) = -V; N(8,3)=-U; N(8,8)=1;

    N(9,1) = U*W;
    N(9,2) = -W; N(9,4)=-U; N(9,9)=1;

    N(10,1)= V*W;
    N(10,3)= -W; N(10,4)=-V; N(10,10)=1;
    N(11,1) = -U^3;
    N(11,2) =  3*U^2;
    N(11,5) = -U;
    N(11,6) = -2*U;
    N(11,11)= 1;

    N(12,1) = -V^3;
    N(12,3) =  3*V^2;
    N(12,5) = -V;
    N(12,7) = -2*V;
    N(12,12)= 1;

    N(13,1) = -W^3;
    N(13,4) =  3*W^2;
    N(13,5) = -W;
    N(13,13)= 1;
    N(14,1)= -U*(V^2-W^2);
    N(14,2)=  V^2-W^2;
    N(14,7)= -U;
    N(14,14)=1;

    N(15,1)= -V*(W^2-U^2);
    N(15,3)=  W^2-U^2;
    N(15,6)=  V;
    N(15,15)=1;

    N(16,1)= -W*(U^2-V^2);
    N(16,4)=  U^2-V^2;
    N(16,6)= -W;
    N(16,16)=1;
    N(17,1) = U^4+V^4+W^4;
    N(17,5) = -2*(U^2+V^2+W^2);
    N(17,17)= 1;

    N(18,1) = U^2*(V^2-W^2);
    N(18,6) = -(V^2-W^2);
    N(18,18)= 1;

    N(19,1) = U^2*(V^2+W^2)-2*V^2*W^2;
    N(19,6) = -(V^2+W^2);
    N(19,7) =  2*V^2;
    N(19,19)= 1;

else
    T = M;
    N = eye(19,19);
end
%% Compute central moments
K_eq = simplify(T*feq);
K_force_guo = simplify(T*force_guo);
K_force = subs(K_force_guo, U, 0);
K_force = subs(K_force, V, 0);
K_force = subs(K_force, W, 0);
Id = eye(19,19);
K_pre = sym(zeros(19,1));
syms k5_pre k6_pre k7_pre k8_pre k9_pre k24_pre k25_pre k26_pre real
K_pre(1) = R;
K_pre(6) = k5_pre;
K_pre(7) = k6_pre;
K_pre(8) = k7_pre;
K_pre(9) = k8_pre;
K_pre(10) = k9_pre;
%post-collision central moments
K_star = (Id-L)*K_pre + L*K_eq;
if(forcing)
    K_star = K_star+ (Id-L/2)*K_force;
end
if(noise)
    K_star = K_star+ Xi;
end
K_star
%post collision populations
syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 real
K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18];
for i=1:19
    if(K_star(i)~=sym(0))
        K_star(i) = K_sym(i);
    end
end
f_post_collision_onestep = collect(simplify(T\K_star), K_star);

%% Fei's two-steps approach
raw_moments = collect(simplify(N^(-1)*K_star), K_star)
syms r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 real
r = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18]'; %symbolic raw moments
f_post_collision_twosteps = simplify(M\r)