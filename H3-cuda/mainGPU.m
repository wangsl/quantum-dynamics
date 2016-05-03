  
% $Id$

function [] = mainGPU(jRot, nVib)

%clear all
close all
clc

format long

addpath(genpath('/home/wang/matlab/quantum-dynamics/common-cuda'));
addpath(genpath('/home/wang/matlab/quantum-dynamics/build'));

if nargin == 0 
  clear all
  clc
  format long;
  jRot = 0;
  nVib = 0;
  addpath(genpath('/home/wang/matlab/quantum-dynamics/common-cuda'));
  addpath(genpath('/home/wang/matlab/quantum-dynamics/build'));
  addpath(genpath('/home/wang/matlab/quantum-dynamics/H3'));
end

global UseLSTH
global H2eV 
global H3Data

%setenv('OMP_NUM_THREADS', '20');

% UseLSTH = true;

MassAU = 1.822888484929367e+03;

mH = 1.007825*MassAU;
mD = 2.01410178*MassAU;
mT = 3.0160492*MassAU;

masses = [ mH mH mH ];
% masses = [ mD mH mH ];

vH2Min = -0.174495770896975;

H2eV = 27.21138505;

% time

CPU = 0;

time.total_steps = int32(100000);
time.time_step = 1;
time.steps = int32(0);

% r1: R

r1.n = int32(2048);
r1.r = linspace(0.4, 16.0, r1.n);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+masses(3));
r1.r0 = 11.0;
r1.k0 = 0.5;
r1.delta = 0.12;

dump1.Cd = 3.0;
dump1.xd = 14.0;
dump1.dump = WoodsSaxon(dump1.Cd, dump1.xd, r1.r);

% r2: r

r2.n = int32(2048);
r2.r = linspace(0.4, 14.0, r2.n);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

% dump functions

dump2.Cd = 3.0;
dump2.xd = 12.0;
dump2.dump = WoodsSaxon(dump2.Cd, dump2.xd, r2.r);

% dividing surface

rd = 8.0;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.8f\n', r2Div);

% angle:

dimensions = 2;

if dimensions == 2 
  % for 2 dimensional case
  theta.n = int32(1);
  theta.m = int32(0);
  theta.x = 1.0;
  theta.w = 2.0;
else 
  % for 3 dimensional case
  theta.n = int32(190);
  theta.m = int32(190);
  [ theta.x, theta.w ] = GaussLegendre(theta.n);
end
  
theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'C2Matlab.m';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);
options.steps_to_copy_psi_from_device_to_host = int32(1000);

% setup potential energy surface and initial wavepacket
pot = H3PESJacobi(r1.r, r2.r, acos(theta.x), masses);

%jRot = 0;
%nVib = 1;
[ psi, eH2, psiH2 ] = InitWavePacket(r1, r2, theta, jRot, nVib);

PlotPotWave(r1, r2, pot, psi);
%return

% cummulative reaction probabilities

CRP.eDiatomic = eH2;
CRP.n_dividing_surface = int32(nDivdSurf);
CRP.n_gradient_points = int32(51);
CRP.n_energies = int32(400);
eLeft = 0.1/H2eV + eH2;
%eLeft = 0.001/H2eV; % + eH2;
eRight = 2.0/H2eV + eH2;
CRP.energies = linspace(eLeft, eRight, CRP.n_energies);
CRP.eta_sq = EtaSq(r1, CRP.energies-eH2);
CRP.CRP = zeros(size(CRP.energies));
CRP.calculate_CRP = int32(1);

% wrapper data to one structure

H3Data.r1 = r1;
H3Data.r2 = r2;
H3Data.theta = theta;
H3Data.pot = pot;
H3Data.psi = psi;
H3Data.time = time;
H3Data.options = options;
H3Data.dump1 = dump1;
H3Data.dump2 = dump2;
H3Data.CRP = CRP;

% time evolution

%CPU = 1
if CPU == 1 
  fprintf('\n == CPU test == \n\n');
  tic
  TimeEvolutionMex(H3Data);
  toc
  return
end

fprintf('\n == GPU test == \n\n');

tic
TimeEvolutionMexCUDA(H3Data);
toc

tic
%psi = H3Data.psi;
PSI = psi(1:2:end, :, :) + j*psi(2:2:end, :, :);
a = sum(sum(conj(PSI).*pot.*PSI));
a = reshape(a, [numel(a), 1]);
sum(theta.w.*a)*r1.dr*r2.dr
toc
