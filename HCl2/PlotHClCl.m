
clear all
%lose all
clc
format long

B2A = 0.52917721092;

RHCl =  1.00:0.01:16.00;
RClCl = 2.60:0.01:16.00;

[ R1, R2 ] = meshgrid(RHCl, RClCl);

theta = 180.0; %75.03;
theta = 100.0;
theta = 2.0;
R3 = sqrt(R1.*R1 + R2.*R2 - 2*R1.*R2.*cos(theta/180*pi));

V = HCl2GHNS(R1, R2, R3);

[C, h] = contour(R1, R2, V, [-8.0:0.20:2.0]./27.2116);
%clabel(C, h);
set(h, 'LineWidth', 1);
set(h, 'LineColor', 'black');
axis normal
