
% Ref: J. Chem. Phys. 110, 11221 (1999)

function [ eta2 ] = EtaSq(R, E)

delta = R.delta;
k0 = R.k0;
m = R.mass;

% translational energy

kE = sqrt(2*m*E);

eta = (pi)^(1/4)*(2*m*delta./kE).^(1/2).*exp(-delta*delta/2*(kE- ...
						  k0).^2);
eta2 = abs(eta).^2;

return
