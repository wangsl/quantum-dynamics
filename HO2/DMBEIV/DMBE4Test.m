
%% To test energies of some special points as in paper
%% J. Phys. Chem. 94, 8073 (1990)

clear all
clc
format long
close all
%addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
%addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

DMBEIVMex(5.663, 1.842, 3.821)

DMBEIVMex(2.282, 7.547, 9.829)

DMBEIVMex(2.806, 2.271, 2.271)

R1=2.5143;
R2=1.8345;
R3=sqrt(R1^2+R2^2-2*R1*R2*cos(104.29/180*pi));
DMBEIVMex(R1, R2, R3)

