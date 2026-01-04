clear; clc;

% ---- input ----
fname = 'vel.txt';     % your file: no header
% If your file is csv-like, use: data = readmatrix(fname);
data = readmatrix(fname);   % robust for whitespace or comma-separated numeric files

% ---- choose columns (edit if vel.txt includes x,y,z first) ----
% Case A: vel.txt = [ux uy uz]
colx = 1; coly = 2; colz = 3;

% Case B (common): vel.txt = [x y z ux uy uz]
% colx = 4; coly = 5; colz = 6;

ux = data(:,colx);
uy = data(:,coly);
uz = data(:,colz);

N = numel(ux);

% ---- moments ----
mux = mean(ux);  muy = mean(uy);  muz = mean(uz);
sigx = std(ux,1); sigy = std(uy,1); sigz = std(uz,1);   % population std (1/N)

fprintf('N = %d\n', N);
fprintf('ux: mean = %+ .6e, std = %.6e\n', mux, sigx);
fprintf('uy: mean = %+ .6e, std = %.6e\n', muy, sigy);
fprintf('uz: mean = %+ .6e, std = %.6e\n', muz, sigz);

% ---- settings ----
nbins = 80;

% =========================
% 1) RAW PDFs
% =========================
figure; hold on; box on;

edges_raw = linspace(min([ux;uy;uz]), max([ux;uy;uz]), nbins+1);

pdfx = histcounts(ux, edges_raw, 'Normalization','pdf');
pdfy = histcounts(uy, edges_raw, 'Normalization','pdf');
pdfz = histcounts(uz, edges_raw, 'Normalization','pdf');

centers_raw = 0.5*(edges_raw(1:end-1)+edges_raw(2:end));

plot(centers_raw, pdfx, 'LineWidth', 1.5);
plot(centers_raw, pdfy, 'LineWidth', 1.5);
plot(centers_raw, pdfz, 'LineWidth', 1.5);

xlabel('u_\alpha');
ylabel('P(u_\alpha)');
legend('u_x','u_y','u_z','Location','best');
title('Velocity PDFs (raw)');
grid on;

% =========================
% 2) NORMALISED PDFs: (u - mean)/std
% =========================
figure; hold on; box on;

uxn = (ux - mux)/sigx;
uyn = (uy - muy)/sigy;
uzn = (uz - muz)/sigz;

edges = linspace(-5, 5, nbins+1);
pdfxN = histcounts(uxn, edges, 'Normalization','pdf');
pdfyN = histcounts(uyn, edges, 'Normalization','pdf');
pdfzN = histcounts(uzn, edges, 'Normalization','pdf');

centers = 0.5*(edges(1:end-1)+edges(2:end));

plot(centers, pdfxN, 'LineWidth', 1.5);
plot(centers, pdfyN, 'LineWidth', 1.5);
plot(centers, pdfzN, 'LineWidth', 1.5);

% Gaussian reference N(0,1)
xx = linspace(-5,5,1000);
gauss = 1/sqrt(2*pi) * exp(-0.5*xx.^2);
plot(xx, gauss, '--', 'LineWidth', 1.5);

xlabel('u_\alpha/\sigma_\alpha');
ylabel('P(u_\alpha/\sigma_\alpha)');
legend('u_x','u_y','u_z','Gaussian','Location','best');
title('Velocity PDFs (normalised)');
grid on;

% ---- optional: save high-quality PDF ----
set(gcf,'Color','w');
exportgraphics(gcf, 'vel_pdf_norm.pdf', 'ContentType','vector');
