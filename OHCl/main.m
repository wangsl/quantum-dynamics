

function [] = main(jRot, nVib)

clear all
clc
format long

if nargin == 0 
  jRot = 0;
  nVib = 0;
end

%clear all
%close all
%clc

%format long

global H2eV 
global OHClData

H2eV = 27.21138505;
vHClMin = -0.1697;

%addpath('/home/wang/matlab/h3-quantum-dynamics', '-end')
addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mCl = 35.453;
mO = 15.999;

masses = [ mO, mH, mCl ];

masses = masses*MassAU;

% time

time.total_steps = int32(15000);
time.time_step = 10;
time.steps = int32(0);

% r1: R

r1.n = int32(768);
r1.r = linspace(1.5, 14.0, r1.n);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+masses(3));
r1.r0 = 10.0;
r1.k0 = 8.0;
r1.delta = 0.035;

eGT = 1/(2*r1.mass)*(r1.k0^2 + 1/(2*r1.delta^2))*H2eV

dump1.Cd = 4.0;
dump1.xd = 12.5;
dump1.dump = WoodsSaxon(dump1.Cd, dump1.xd, r1.r);
%dump1.dump(:) = 1.0;

% r2: r

r2.n = int32(512);
r2.r = linspace(0.4, 10.0, r2.n);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

dump2.Cd = 4.0;
dump2.xd = 8.0;
dump2.dump = WoodsSaxon(dump2.Cd, dump2.xd, r2.r);
%dump2.dump(:) = 1.0;

% dividing surface

rd = 5.5;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.8f\n', r2Div);

% theta

theta.n = int32(199);
theta.m = int32(180);
[ theta.x, theta.w ] = GaussLegendre(theta.n);

%acos(theta.x)/pi*180
%return

theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'OHClMatlab.m';
%options.CRPMatFile = 'CRPMat.mat';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);

% setup potential energy surface and initial wavepacket
pot = OHClPESJacobi(r1.r, r2.r, acos(theta.x), masses);

%jRot = 0;
%nVib = 0;
[ psi, eHCl, psiHCl ] = InitWavePacket(r1, r2, theta, jRot, nVib);

%PlotPotWave(r1, r2, pot, psi)

% cummulative reaction probabilities

CRP.eDiatomic = eHCl;
CRP.n_dividing_surface = nDivdSurf;
CRP.n_gradient_points = int32(11);
CRP.n_energies = int32(150);
eLeft = 0.01/H2eV + eHCl;
eRight = 1.5/H2eV + eHCl;
CRP.energies = linspace(eLeft, eRight, CRP.n_energies);
CRP.eta_sq = EtaSq(r1, CRP.energies-eHCl);
CRP.CRP = zeros(size(CRP.energies));
CRP.calculate_CRP = int32(1);

%plot(CRP.energies*H2eV, CRP.eta_sq)
%return

% wrapper data to one structure

OHClData.r1 = r1;
OHClData.r2 = r2;
OHClData.theta = theta;
OHClData.pot = pot;
OHClData.psi = psi;
OHClData.time = time;
OHClData.options = options;
OHClData.dump1 = dump1;
OHClData.dump2 = dump2;
OHClData.CRP = CRP;

% time evolution

tic
TimeEvolutionMex(OHClData);
toc

return


