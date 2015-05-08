
clear all
%lose all
clc
format long

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mCl = 35.453;

masses = [ mH, mCl, mCl ];

R = 3.0:0.01:14.0;
r = 2.8:0.01:9.0;

theta = 0.0;
theta = [ theta/180*pi ];

V = HCl2PESJacobi(R, r, theta, masses);

[C, h] = contour(R, r, V', [-0.3:0.005:0.1]);
%clabel(C, h);
set(h, 'LineWidth', 1.2);
set(h, 'LineColor', 'black');
axis normal
