
% $Id$

function [] = C2Matlab() 

global H3Data

fprintf(' From C2Matlab\n')

if mod(H3Data.time.steps, 20) == 0
  PlotPotWave(H3Data.r1, H3Data.r2, H3Data.pot, H3Data.psi)
end

return

if mod(H3Data.time.steps, 100) == 0
  if H3Data.CRP.calculate_CRP == 1
    CRP = H3Data.CRP;
    save(H3Data.options.CRPMatFile, 'CRP');
  end
end

return

if mod(H3Data.time.steps, 20) == 0
  PlotPotWave(H3Data.r1, H3Data.r2, H3Data.pot, H3Data.psi)
end

return

