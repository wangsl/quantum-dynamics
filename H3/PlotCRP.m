
function [] = PlotCRP(jRot, nVib)

if nargin == 0 
  jRot = 0;
  nVib = 0;
end


%clear all
%close all
%clc

%format long

H2eV = 27.21138505;

CRPMatFile = sprintf('CRPMat-j%d-v%d.mat', jRot, nVib)

load(CRPMatFile);

E = CRP.energies;
crp = CRP.CRP;

h_CRP = plot(E*H2eV, crp, 'b', 'LineWidth', 2.5, ...
	     'MarkerEdgeColor','r', ...
	     'MarkerFaceColor', 'y', 'MarkerSize', 3.5, ...
	     'YDataSource', 'crp');

grid on;

set(gca, 'xtick', [0.4:0.4:max(E)*H2eV]);
set(gca, 'ytick', [0.0:0.1:1.2]);

set(gca,'FontSize',12,'LineWidth',1.25)
%set(gca,'XMinorTick','on','YMinorTick','on');

hold off

axis([min(E)*H2eV, max(E)*H2eV, -0.1, 1.1]);

title(['Initial State Selected Reaction Probabilities for H + H_2' ...
       sprintf(' (j=%d, v=%d)', jRot, nVib) ], ...
      'FontSize', 14)
xlabel('Total Energy (eV)', 'FontSize', 14)
ylabel('Reaction Probabilities', 'FontSize', 14)

hold off

