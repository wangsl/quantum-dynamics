

clear all
clc
format long
%close all

rOO = linspace(1.65, 8.0, 512);
V = OOPES(rOO);


plot(rOO, V, 'b', 'LineWidth', 3);


%vOOMin = 0; %-0.191570045;

%R3 = 200;
%myF = @(x) DMBEIVMex(x, x+R3, R3) - vOOMin

myF = @(x) OOPES(x);

options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');

[ r, v ] = fminunc(myF, [2.0], options)