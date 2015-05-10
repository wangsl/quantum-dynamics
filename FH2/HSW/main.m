
%clear all
%clc
format long

addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

setenv('HSW_DATA_DIR', ...
       '/home/wang/matlab/quantum-dynamics/FH2/HSW')

n = 512*512*200;
r1 = zeros(n, 1);
r2 = zeros(n, 1);
r3 = zeros(n, 1);

r1(:) = 1.2;
r2(:) = 2.5;
r3(:) = 2.8;

V = HSWMex(r1, r2, r3);
