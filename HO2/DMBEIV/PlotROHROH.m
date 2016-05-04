
clear all
clc
format long
%close all
%addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
%addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

R2 = linspace(1.0, 8.0, 512);
R3 = linspace(1.0, 8.0, 512);

[ R2, R3 ]  = meshgrid(R2, R3);

R1 = R2 + R3;

V = DMBEIVPES(R1, R2, R3);

[ C, h ] = contour(R2, R3, V, [-0.2:0.01:0.2]);

set(h, 'LineWidth', 2);
%set(h, 'LineColor', 'black');
%axis normal;
axis square;

min(min(V))
