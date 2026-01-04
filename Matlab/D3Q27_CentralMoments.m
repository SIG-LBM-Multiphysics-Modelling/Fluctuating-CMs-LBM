clear all
clc

central_moments = true;

syms U V W R Fx Fy Fz omega k4_star k5_star k6_star k7_star k8_star...
     f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26...
     kBT eta4 eta5 eta6 eta7 eta8 eta9 eta10 eta11 eta12 eta13 eta14 eta15 eta16 eta17 eta18 eta19 eta20 eta21 eta22 eta23 eta24 eta25 eta26 real
f = [f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26];
eta = [0 0 0 0 eta4 eta5 eta6 eta7 eta8 eta9 eta10 eta11 eta12 eta13 eta14 eta15 eta16 eta17 eta18 eta19 eta20 eta21 eta22 eta23 eta24 eta25 eta26]';
L = diag([1, 1, 1, 1, 1, 1, 1, omega, omega, omega, 1, 1, 1, 1,...
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
cs = 1/sqrt(3);
cs2 = cs^2;
cs4 = cs^4;
cs6 = cs^6;
cs8 = cs^8;
cs10 = cs^10;
cs12 = cs^12;

cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1];
cy = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1];
cz = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1];
w = [8./27., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,... 
	1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,...
	1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216.];
u2 = U*U+V*V+W*W;
Fy = 0;
Fx = 0;
Fz = 0;
T = sym(zeros(length(cx),length(cx)));
M = zeros(length(cx),length(cx));
feq = sym(zeros(length(cx),1));
%choose the maximum order of the Hermite polynomials N=2,3,4,5,6
for i=1:length(cx)
    first_order = U*cx(i)+V*cy(i)+W*cz(i);
    first_order = first_order/cs2;
    
    second_order = 4.5*(U*cx(i)+V*cy(i)+W*cz(i))^2-1.5*u2;
    
    third_order = (cx(i)^2-cs2)*cy(i)*U*U*V + (cx(i)^2-cs2)*cz(i)*U*U*W +...
                  (cy(i)^2-cs2)*cx(i)*U*V*V + (cz(i)^2-cs2)*cx(i)*U*W*W +...
                  (cz(i)^2-cs2)*cy(i)*V*W*W + (cy(i)^2-cs2)*cz(i)*V*V*W +...
              2*( cx(i)*cy(i)*cz(i)*U*V*W);  
    third_order = third_order/(2*cs6);

    fourth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*U*U*V*V +...
                   (cx(i)^2-cs2)*(cz(i)^2-cs2)*U*U*W*W +...
                   (cy(i)^2-cs2)*(cz(i)^2-cs2)*V*V*W*W +...
                2*( cx(i)*cy(i)*(cz(i)^2-cs2)*U*V*W*W +...
                    cx(i)*(cy(i)^2-cs2)*cz(i)*U*V*V*W +...
                   (cx(i)^2-cs2)*cy(i)*cz(i)*U*U*V*W );
    fourth_order = fourth_order/(4*cs8);
    
    fifth_order = (cx(i)^2-cs2)*cy(i)*(cz(i)^2-cs2)*U*U*V*W*W +...
                  (cx(i)^2-cs2)*(cy(i)^2-cs2)*cz(i)*U*U*V*V*W +...
                  cx(i)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*V*V*W*W;
    fifth_order = fifth_order/(4*cs10);

    sixth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*U*V*V*W*W;
    sixth_order = sixth_order/(8*cs12);
    feq(i) = R*w(i)*(1+first_order + second_order + third_order +...
                         fourth_order + fifth_order + sixth_order);
    A = U*cx(i)+V*cy(i)+W*cz(i);
    force(i) = w(i)*( Fx*(3*(cx(i)-U)+9*A*cx(i))+...
                          Fy*(3*(cy(i)-V)+9*A*cy(i))+...
						  Fz*(3*(cz(i)-W)+9*A*cz(i)) );
    % Set the basis
    CX = cx(i)-U;
    CY = cy(i)-V;
    CZ = cz(i)-W;
    CX2 = CX*CX;
    CY2 = CY*CY;
    CZ2 = CZ*CZ;
    T(1,i) = 1;
    T(2,i) = CX;
    T(3,i) = CY;
    T(4,i) = CZ;
    T(5,i) = CX2-cs2;
    T(6,i) = CY2-cs2;
    T(7,i) = CZ2-cs2;
    T(8,i) = CX*CY;
    T(9,i) = CX*CZ;
    T(10,i) = CY*CZ;
    T(11,i) = CX2*CY-cs2*CY;
    T(12,i) = CX2*CZ-cs2*CZ;
    T(13,i) = CX*CY2-cs2*CX;
    T(14,i) = CX*CZ2-cs2*CX;
    T(15,i) = CY*CZ2-cs2*CY;
    T(16,i) = CY2*CZ-cs2*CZ;
    T(17,i) = CX*CY*CZ;
    T(18,i) = CX2*CY2 - cs2*(CX2+CY2) + cs4;
    T(19,i) = CX2*CZ2 - cs2*(CX2+CZ2) + cs4;
    T(20,i) = CY2*CZ2 - cs2*(CY2+CZ2) + cs4;
    T(21,i) = CX*CY*CZ2 - cs2*CX*CY;
    T(22,i) = CX*CY2*CZ - cs2*CX*CZ;
    T(23,i) = CX2*CY*CZ - cs2*CY*CZ;
    T(24,i) = CX2*CY*CZ2 - cs2*(CX2*CY+CY*CZ2) + cs4*CY;
    T(25,i) = CX2*CY2*CZ - cs2*(CX2*CZ+CY2*CZ) + cs4*CZ;
    T(26,i) = CX*CY2*CZ2 - cs2*(CX*CY2+CX*CZ2) + cs4*CX;
    T(27,i) = CX2*CY2*CZ2 - cs2*(CX2*CY2+CX2*CZ2+CY2*CZ2) + cs4*(CX2+CY2+CZ2) - cs6;
    
    CX = cx(i);
    CY = cy(i);
    CZ = cz(i);
    CX2 = CX*CX;
    CY2 = CY*CY;
    CZ2 = CZ*CZ;
    M(1,i) = 1;
    M(2,i) = CX;
    M(3,i) = CY;
    M(4,i) = CZ;
    M(5,i) = CX2-cs2;
    M(6,i) = CY2-cs2;
    M(7,i) = CZ2-cs2;
    M(8,i) = CX*CY;
    M(9,i) = CX*CZ;
    M(10,i) = CY*CZ;
    M(11,i) = CX2*CY-cs2*CY;
    M(12,i) = CX2*CZ-cs2*CZ;
    M(13,i) = CX*CY2-cs2*CX;
    M(14,i) = CX*CZ2-cs2*CX;
    M(15,i) = CY*CZ2-cs2*CY;
    M(16,i) = CY2*CZ-cs2*CZ;
    M(17,i) = CX*CY*CZ;
    M(18,i) = CX2*CY2 - cs2*(CX2+CY2) + cs4;
    M(19,i) = CX2*CZ2 - cs2*(CX2+CZ2) + cs4;
    M(20,i) = CY2*CZ2 - cs2*(CY2+CZ2) + cs4;
    M(21,i) = CX*CY*CZ2 - cs2*CX*CY;
    M(22,i) = CX*CY2*CZ - cs2*CX*CZ;
    M(23,i) = CX2*CY*CZ - cs2*CY*CZ;
    M(24,i) = CX2*CY*CZ2 - cs2*(CX2*CY+CY*CZ2) + cs4*CY;
    M(25,i) = CX2*CY2*CZ - cs2*(CX2*CZ+CY2*CZ) + cs4*CZ;
    M(26,i) = CX*CY2*CZ2 - cs2*(CX*CY2+CX*CZ2) + cs4*CX;
    M(27,i) = CX2*CY2*CZ2 - cs2*(CX2*CY2+CX2*CZ2+CY2*CZ2) + cs4*(CX2+CY2+CZ2) - cs6;
end
% Compute central moments
if(central_moments)
    T = simplify(T);
    N = simplify(T/M); %shift matrix
else
    T = M;
    N = eye(27,27);
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

K_eq = simplify(T*feq);
K_force = simplify(T*force');
Id = eye(length(cx),length(cx));
K_pre = sym(zeros(length(cx),1));
syms k4_pre k5_pre k6_pre k7_pre k8_pre k9_pre real
K_pre(1) = R;
K_pre(8) = k7_pre;
K_pre(9) = k8_pre;
K_pre(10) = k9_pre;
%post-collision central moments
K_star = (Id-L)*K_pre + L*K_eq + (Id-L/2)*K_force + Xi


%post collision populations
syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19...
     k20 k21 k22 k23 k24 k25 k26 real
K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19...
         k20 k21 k22 k23 k24 k25 k26];
for i=1:length(cx)
    if(K_star(i)~=sym(0))
        K_star(i) = K_sym(i);
    end
end
% f_post_collision_onestep = collect(simplify(T\K_star), K_star); 
%% Fei's two-steps approach
raw_moments = collect(simplify(N \ K_star), K_star)
syms r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26 real
r = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26]'; %symbolic raw moments
f_post_collision_twosteps = collect(simplify(M\r),K_star);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/6);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/6);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/18);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/18);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/24);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/24);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/12);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/12);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/72);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/72);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 2/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -2/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -4/9);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/9);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/9)
% %%
% % 
% % 
% % 
% % K_eq = simplify(T'*feq');
% % Id = eye(27,27);
% % K_pre = sym(zeros(27,1));
% % K_pre(1) = R;
% % K_pre(5) = k4_star;
% % K_pre(6) = k5_star;
% % K_pre(7) = k6_star;
% % K_pre(8) = k7_star;
% % K_pre(9) = k8_star;
% % 
% % K_star = (Id-L)*K_pre + L*K_eq
% % syms k4 k5 k6 k7 k8
% % K_star(5) = k4;
% % K_star(6) = k5;
% % K_star(7) = k6;
% % K_star(8) = k7;
% % K_star(9) = k8;
% % % syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18...
% % %      k19 k20 k21 k22 k23 k24 k25 k26 real
% % % K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17...
% % %          k18 k19 k20 k21 k22 k23 k24 k25 k26];
% % % for i=1:27
% % %     if(K_star(i)~=sym(0))
% % %         K_star(i) = K_sym(i);
% % %     end
% % % end
% % f_post_collision = simplify((T')^(-1) * K_star);
% % f_post_collision = collect(f_post_collision, [R K_star])
% % fid = fopen('post_collision.txt', 'w');
% % for i=1:length(f_post_collision)
% %     fprintf(fid,'post_collision_f[%d] = %s;\n', i-1, f_post_collision(i));
% % end
% % fclose(fid);
% 
% fid = fopen('NMatrix.txt', 'w');
% for i=1:27
%     for j=1:27
%         if(N(i,j) ~= 0)
%             fprintf(fid,'  N(%d,%d) = %s;  ', i-1, j-1, N(i,j));
%         end
%     end
%     fprintf(fid,' \n');
% end
% fclose(fid);
% 
% fid = fopen('MMatrix.txt', 'w');
% for i=1:27
%     for j=1:27
%         if(M(i,j) ~= 0)
%             fprintf(fid,'  M(%d,%d) = %g;  ', i-1, j-1, M(i,j));
%         end
%     end
%     fprintf(fid,' \n');
% end
% fclose(fid);
