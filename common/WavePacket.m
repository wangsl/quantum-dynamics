
% $Id$

function [ psiReal, psiImag ] = WavePacket(psi)

psiReal = psi(1:2:end, :, :);
psiImag = psi(2:2:end, :, :);

return

