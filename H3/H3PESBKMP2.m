
% $Id$

% Ref: J. Chem. Phys. 104. 7139 (1996)

function [ varargout ] = H3PESBKMP2(r1, r2, r3)

persistent first

if isempty(first) 
  fprintf(' To use BKMP2 PES\n');
  first = 0;
end

% r1, r2 and r3 should be one dimensional array

R1 = reshape(r1, [numel(r1), 1]);
R2 = reshape(r2, [numel(r2), 1]);
R3 = reshape(r3, [numel(r3), 1]);

vH2Min = -0.174495770896975;

if nargout == 0 | nargout == 1
  varargout{1} = BKMP2Mex(R1, R2, R3);
elseif nargout == 2
  [ varargout{1}, varargout{2} ] = BKMP2Mex(R1, R2, R3);
end

varargout{1} = varargout{1} - vH2Min;

return
