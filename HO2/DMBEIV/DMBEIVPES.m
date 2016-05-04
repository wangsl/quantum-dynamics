
function [ V ] = DMBEIVPES(R1, R2, R3)

vOOMin = -0.19157004525;

V = DMBEIVMex(reshape(R1, [numel(R1), 1]), ...
	      reshape(R2, [numel(R2), 1]), ...
	      reshape(R3, [numel(R3), 1])) - vOOMin;

V = reshape(V, size(R1));


