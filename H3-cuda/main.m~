  
% $Id$

function [] = main(jRot, nVib)

%clear all
%close all
%clc

format long

if nargin == 0 
  jRot = 0;
  nVib = 0;
end

global UseLSTH
global H2eV 
global H3Data

%addpath(genpath('/home/wang/matlab/quantum-dynamics'));
%addpath(genpath('/home/wang/matlab/quantum-dynamics/H3'));

setenv('OMP_NUM_THREADS', '20');

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

time.total_steps = int32(5000);
time.time_step = 5;
time.steps = int32(0);

% r1: R

r1.n = int32(256);
r1.r = linspace(0.4, 14.0, r1.n);
r1.dr = r1.r(2) - r1.r(1);
r1.mass = masses(1)*(masses(2)+masses(3))/(masses(1)+masses(2)+masses(3));
r1.r0 = 8.0;
r1.k0 = 4.0;
r1.delta = 0.12;

dump1.Cd = 3.0;
dump1.xd = 12.0;
dump1.dump = WoodsSaxon(dump1.Cd, dump1.xd, r1.r);

% r2: r

r2.n = int32(256);
r2.r = linspace(0.4, 10.0, r2.n);
r2.dr = r2.r(2) - r2.r(1);
r2.mass = masses(2)*masses(3)/(masses(2)+masses(3));

% dump functions

dump2.Cd = 3.0;
dump2.xd = 8.0;
dump2.dump = WoodsSaxon(dump2.Cd, dump2.xd, r2.r);

% dividing surface

rd = 5.0;
nDivdSurf = int32((rd - min(r2.r))/r2.dr);
r2Div = double(nDivdSurf)*r2.dr + min(r2.r);
fprintf(' Dviding surface: %.8f\n', r2Div);

% angle:

dimensions = 3;

if dimensions == 2 
  % for 2 dimensional case
  theta.n = int32(1);
  theta.m = int32(0);
  theta.x = 1.0;
  theta.w = 2.0;
else 
  % for 3 dimensional case
  theta.n = int32(120);
  theta.m = int32(100);
  [ theta.x, theta.w ] = GaussLegendre(theta.n);
end
  
theta.legendre = LegendreP2(double(theta.m), theta.x);
% transpose Legendre polynomials in order to do 
% matrix multiplication in C++ and Fortran LegTransform.F
theta.legendre = theta.legendre';

% options

options.wave_to_matlab = 'C2Matlab.m';
options.CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib);

% setup potential energy surface and initial wavepacket
pot = H3PESJacobi(r1.r, r2.r, acos(theta.x), masses);

%jRot = 0;
%nVib = 1;
[ psi, eH2, psiH2 ] = InitWavePacket(r1, r2, theta, jRot, nVib);

% cummulative reaction probabilities

CRP.eDiatomic = eH2;
CRP.n_dividing_surface = nDivdSurf;
CRP.n_gradient_points = int32(11);
CRP.n_energies = int32(400);
eLeft = 0.4/H2eV;
eRight = 4.0/H2eV;
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

tic
TimeEvolutionMex(H3Data);
toc

return

tic
psi = H3Data.psi;
PSI = psi(1:2:end, :, :) + j*psi(2:2:end, :, :);
a = sum(sum(conj(PSI).*PSI));
a = reshape(a, [numel(a), 1]);
sum(theta.w.*a)*r1.dr*r2.dr
toc
