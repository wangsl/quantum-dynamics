
clear all
%lose all
clc
format long

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mCl = 35.453;

masses = [ mH, mCl, mCl ];

ptions = optimoptions('fminunc', 'Algorithm', 'quasi-newton');
options.Display = 'iter';
options.MaxFunEvals = 2000;
options.TolFun = 1.0e-10;
options.TolX = 1.0e-10;

fun = @(r) Cl2PES(r);

[x, fval ] = fminunc(fun, [ 1.0 ], options)

