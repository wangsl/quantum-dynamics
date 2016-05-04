
format long
clear all
clc

R2 = linspace(1.0, 6, 200);
R3 = linspace(1.0, 6, 200);

[ R2, R3 ]  = meshgrid(R2, R3);

%theta = 103.7/180*pi;
%theta = pi;

%R3 = sqrt(R1.^2 + R2.^2 - 2*R1.*R2.*cos(theta));
R1 = R2 + R3;

V = HO2PES(R1, R2, R3);

[ C, h ] = contour(R2, R3, V, [-0.25:0.01:0.02]);

set(h, 'LineWidth', 2);
set(h, 'LineColor', 'black');
%axis normal;
axis square;