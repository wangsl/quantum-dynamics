
function [] = OHClMatlab()

global OHClData

fprintf(' From OHClMatlab\n')

if mod(OHClData.time.steps, 20) == 0
  PlotCRP();
  if OHClData.CRP.calculate_CRP == 1
    CRP = OHClData.CRP;
    save(OHClData.options.CRPMatFile, 'CRP');
  end
end

if mod(OHClData.time.steps, 20) == 0
  PlotPotWave(OHClData.r1, OHClData.r2, OHClData.pot, OHClData.psi)
end

return
