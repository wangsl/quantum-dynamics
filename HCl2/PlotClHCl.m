
clear all
%lose all
clc
format long

RHCl = 0.60:0.01:5.0;

[ R1, R3 ] = meshgrid(RHCl, RHCl);

theta = 136.23;
R2 = sqrt(R1.*R1 + R3.*R3 - 2*R1.*R3.*cos(theta/180*pi));

V = HCl2GHNS(R1, R2, R3);

[C, h] = contour(R1, R3, V,  [-8.0:0.50:5.0]);
%clabel(C, h);
set(h, 'LineWidth', 1.2);
set(h, 'LineColor', 'black');
axis normal
