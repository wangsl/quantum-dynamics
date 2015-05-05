
clear all
%lose all
clc
format long

B2A = 0.52917721092;

RHCl =  0.60:0.01:5.00;
RClCl = 1.40:0.01:5.00;

RHCl = RHCl*B2A;
RClCl = RClCl*B2A;

[ R1, R2 ] = meshgrid(RHCl, RClCl);

theta = 175.03;
R3 = sqrt(R1.*R1 + R2.*R2 - 2*R1.*R2.*cos(theta/180*pi));

V = HCl2GHNS(R1, R2, R3);

[C, h] = contour(R1, R2, V, [-8.0:0.50:5.0]./27.2116);
%clabel(C, h);
set(h, 'LineWidth', 1);
set(h, 'LineColor', 'black');
axis normal
