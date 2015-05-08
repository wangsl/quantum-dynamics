
function [] = HCl2Matlab()

global HCl2Data

fprintf(' From HCl2Matlab\n')

return

if mod(HCl2Data.time.steps, 10) == 0
  PlotCRP();
  if HCl2Data.CRP.calculate_CRP == 1
    CRP = HCl2Data.CRP;
    save(HCl2Data.options.CRPMatFile, 'CRP');
  end
end

if mod(HCl2Data.time.steps, 10) == 0
  PlotPotWave(HCl2Data.r1, HCl2Data.r2, HCl2Data.pot, HCl2Data.psi)
end

return
