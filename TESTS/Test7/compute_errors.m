clear all
clc

A = load('case1.txt');
t = A(:,1);
u2 = A(:,2);
v2 = A(:,3);
w2 = A(:,4);
e = (u2+v2+w2)/3;
target = A(:,5);
case1.Du2 = sum(u2-target)/length(t);
case1.Dv2 = sum(v2-target)/length(t);
case1.Dw2 = sum(w2-target)/length(t);
case1.De = sum(e-target)/length(t);
case1.RMS_u2 = sqrt(sum((u2-target).^2)/length(t));
case1.RMS_v2 = sqrt(sum((v2-target).^2)/length(t));
case1.RMS_w2 = sqrt(sum((w2-target).^2)/length(t));
case1.RMS_e = sqrt(sum((e-target).^2)/length(t));

A = load('case2.txt');
t = A(:,1);
u2 = A(:,2);
v2 = A(:,3);
w2 = A(:,4);
e = (u2+v2+w2)/3;
target = A(:,5);
case2.Du2 = sum(u2-target)/length(t);
case2.Dv2 = sum(v2-target)/length(t);
case2.Dw2 = sum(w2-target)/length(t);
case2.De = sum(e-target)/length(t);
case2.RMS_u2 = sqrt(sum((u2-target).^2)/length(t));
case2.RMS_v2 = sqrt(sum((v2-target).^2)/length(t));
case2.RMS_w2 = sqrt(sum((w2-target).^2)/length(t));
case2.RMS_e = sqrt(sum((e-target).^2)/length(t));

A = load('case3.txt');
t = A(:,1);
u2 = A(:,2);
v2 = A(:,3);
w2 = A(:,4);
e = (u2+v2+w2)/3;
target = A(:,5);
case3.Du2 = sum(u2-target)/length(t);
case3.Dv2 = sum(v2-target)/length(t);
case3.Dw2 = sum(w2-target)/length(t);
case3.De = sum(e-target)/length(t);
case3.RMS_u2 = sqrt(sum((u2-target).^2)/length(t));
case3.RMS_v2 = sqrt(sum((v2-target).^2)/length(t));
case3.RMS_w2 = sqrt(sum((w2-target).^2)/length(t));
case3.RMS_e = sqrt(sum((e-target).^2)/length(t));

A = load('case4.txt');
t = A(:,1);
u2 = A(:,2);
v2 = A(:,3);
w2 = A(:,4);
e = (u2+v2+w2)/3;
target = A(:,5);
case4.Du2 = sum(u2-target)/length(t);
case4.Dv2 = sum(v2-target)/length(t);
case4.Dw2 = sum(w2-target)/length(t);
case4.De = sum(e-target)/length(t);
case4.RMS_u2 = sqrt(sum((u2-target).^2)/length(t));
case4.RMS_v2 = sqrt(sum((v2-target).^2)/length(t));
case4.RMS_w2 = sqrt(sum((w2-target).^2)/length(t));
case4.RMS_e = sqrt(sum((e-target).^2)/length(t));