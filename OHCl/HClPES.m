
function [ V ] = HClPES(rHCl)

rHCl = reshape(rHCl, [ numel(rHCl), 1 ]);

rOCl = zeros(size(rHCl));
rOCl(:) = 100.0;
rOH = sqrt(rHCl.*rHCl + rOCl.*rOCl);

vHClMin = -0.1697;

V = OHClKSGMex(rOH, rOCl, rHCl) - vHClMin;

