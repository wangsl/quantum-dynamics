
function [] = PlotCRP()

global H2eV 
global H3Data

eH2 = H3Data.CRP.eDiatomic;

%E = (H3Data.CRP.energies-eH2)*H2eV;
E = (H3Data.CRP.energies-eH2)*H2eV;
CRP = -H3Data.CRP.CRP;

persistent CRPPlot

if isempty(CRPPlot) 
  
  figure(2)
  
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 3, 'YDataSource', 'CRP');
  
  %hold off
  
  %axis([min(E), max(E), -0.2, 1.0]);
end

refreshdata(CRPPlot, 'caller');
drawnow
