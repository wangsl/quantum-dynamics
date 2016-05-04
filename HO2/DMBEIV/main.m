

function [] = main(jRot, nVib)

%clear all
clc
format long

%warning off MATLAB:maxNumCompThreads:Deprecated
%maxNumCompThreads(10);
%setenv('OMP_NUM_THREADS', '10');

if nargin == 0 
  clear all
  clc
  format long
  jRot = 0;
  nVib = 0;
end
addpath(genpath('/home/wang/matlab/quantum-dynamics/build'))
addpath(genpath('/home/wang/matlab/quantum-dynamics/common'))

global H2eV 
global HO2Data

H2eV = 27.21138505;

MassAU = 1.822888484929367e+03;

mH = 1.0079;
mO = 15.999;

masses = [ mH, mO, mO ];

masses = masses*MassAU;

% time

time.total_steps = int32(500000);
time.time_step = 1;
time.steps = int32(0);

% r1: R

r1.n = int32(960);
r1.r = linspace(1.5, 16.0, r1.n);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+masses(3));
r1.r0 = 10.0;
r1.k0 = 0.25;
r1.delta = 0.06;

eGT = 1/(2*r1.mass)*(r1.k0^2 + 1/(2*r1.delta^2))*H2eV

dump1.Cd = 4.0;
dump1.xd = 14.5;
dump1.dump = WoodsSaxon(dump1.Cd, dump1.xd, r1.r);

% r2: r

r2.n = int32(960);
r2.r = linspace(1.5, 12.0, r2.n);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

dump2.Cd = 4.0;
dump2.xd = 10.0;
dump2.dump = WoodsSaxon(dump2.Cd, dump2.xd, r2.r);

% dividing surface

rd = 7.0;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.8f\n', r2Div);

% theta

theta.n = int32(190);
theta.m = int32(190);
[ theta.x, theta.w ] = GaussLegendre(theta.n);

theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'HO2Matlab.m';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);
options.steps_to_copy_psi_from_device_to_host = int32(50);

% setup potential energy surface and initial wavepacket
pot = DMBEIVPESJacobi(r1.r, r2.r, acos(theta.x), masses);

[ psi, eO2, psiO2 ] = InitWavePacket(r1, r2, theta, jRot, nVib);

PlotPotWave(r1, r2, pot, psi)

% cummulative reaction probabilities

CRP.eDiatomic = eO2;
CRP.n_dividing_surface = nDivdSurf;
CRP.n_gradient_points = int32(31);
CRP.n_energies = int32(300);
eLeft = 0.6/H2eV + eO2;
eRight = 1.8/H2eV + eO2;
CRP.energies = linspace(eLeft, eRight, CRP.n_energies);
CRP.eta_sq = EtaSq(r1, CRP.energies-eO2);
CRP.CRP = zeros(size(CRP.energies));
CRP.calculate_CRP = int32(1);

% pack data to one structure

HO2Data.r1 = r1;
HO2Data.r2 = r2;
HO2Data.theta = theta;
HO2Data.pot = pot;
HO2Data.psi = psi;
HO2Data.time = time;
HO2Data.options = options;
HO2Data.dump1 = dump1;
HO2Data.dump2 = dump2;
HO2Data.CRP = CRP;

% time evolution

tic
%TimeEvolutionMex(HO2Data);
TimeEvolutionMexCUDA(HO2Data);
toc

return

