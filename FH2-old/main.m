
%clear all
%clc
format long

addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

n = 256*256*64; %*200;
r1 = zeros(n, 1);
r2 = zeros(n, 1);
r3 = zeros(n, 1);

r1(:) = 1.2;
r2(:) = 2.5;
r3(:) = 2.8;

V = FXZMex(r1, r2, r3);