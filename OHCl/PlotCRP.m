
function [] = PlotCRP()

global H2eV 
global OHClData

eHCl = OHClData.CRP.eDiatomic;

E = (OHClData.CRP.energies-eHCl)*H2eV;
CRP = OHClData.CRP.CRP;

persistent CRPPlot

if isempty(CRPPlot) 
  
  figure(2)
  
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 3, 'YDataSource', 'CRP');
  
  %hold off
  
  %axis([min(E), max(E), -0.2, 1.0]);
end

refreshdata(CRPPlot, 'caller');
drawnow
