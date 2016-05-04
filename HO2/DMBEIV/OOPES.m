
function [ V ] = OOPES(rOO)

r1 = reshape(rOO, [numel(rOO), 1]);

r3 = zeros(size(r1));
r3(:) = 100.0;

r2 = r1 + r3;

vOOMin = -0.19157004525;

V = DMBEIVMex(r1, r2, r3) - vOOMin;
