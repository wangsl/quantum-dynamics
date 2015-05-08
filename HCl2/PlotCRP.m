
function [] = PlotCRP()

global H2eV 
global HCl2Data

eHCl = HCl2Data.CRP.eDiatomic;

E = (HCl2Data.CRP.energies-eHCl)*H2eV;
CRP = HCl2Data.CRP.CRP;

persistent CRPPlot

if isempty(CRPPlot) 
  
  figure(2)
  
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 3, 'YDataSource', 'CRP');
end

refreshdata(CRPPlot, 'caller');
drawnow
