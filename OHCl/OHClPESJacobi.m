
% $Id$

function [ V ] = OHClPESJacobi(R, r, Theta, masses)

vHClMin = -0.1697;

f2 = masses(3)/(masses(2)+masses(3));
f3 = 1 - f2;

nR = length(R);
nr = length(r);
nTheta = length(Theta);

[x, y, z] = meshgrid(R, r, Theta);

r23 = y;
r12 = sqrt((f2*y).^2 + x.^2 - 2*f2*y.*x.*cos(z));
r13 = sqrt((f3*y).^2 + x.^2 + 2*f3*y.*x.*cos(z));

r23 = reshape(r23, [numel(r23), 1]);
r12 = reshape(r12, [numel(r12), 1]);
r13 = reshape(r13, [numel(r13), 1]);

V = OHClKSGMex(r12, r13, r23) - vHClMin;

V = reshape(V, [nr, nR, nTheta]);

V = permute(V, [2 1 3]);

return
