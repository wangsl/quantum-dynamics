
function [] = PlotCRP()

global H2eV 
global HO2Data

eO2 = HO2Data.CRP.eDiatomic;

E = (HO2Data.CRP.energies-eO2)*H2eV;
CRP = -2*HO2Data.CRP.CRP;

persistent CRPPlot

if isempty(CRPPlot) 
  figure(2)
  CRPPlot = plot(E, CRP, 'b-', 'LineWidth', 3, 'YDataSource', 'CRP');
end

refreshdata(CRPPlot, 'caller');
drawnow
