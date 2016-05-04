
clear all
clc
format long
%addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
%addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

R1 = linspace(1.5, 7.5, 512);
R2 = linspace(1.0, 7.0, 512);

[ R1, R2 ]  = meshgrid(R1, R2);
theta = 103.7/180*pi;
R3 = sqrt(R1.^2 + R2.^2 - 2*R1.*R2*cos(theta));

V = DMBEIVPES(R1, R2, R3);

[ C, h ] = contour(R1, R2, V, [-0.15:0.01:0.2]);

set(h, 'LineWidth', 1.5);
%set(h, 'LineColor', 'black');
%axis normal;
axis square;
colormap(hsv)

min(min(V))