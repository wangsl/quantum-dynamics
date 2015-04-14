
function [ psi, eH2, psiH2 ] = InitWavePacket(R1, R2, Theta, jRot, nVib)

n1 = R1.n;
n2 = R2.n;
nTheta = Theta.n;

r1 = R1.r;
delta = R1.delta;
r10 = R1.r0;
k0 = R1.k0;

g = (1/(pi*delta^2))^(1/4) * ...
    exp(-(r1-r10).^2/(2*delta*delta) - j*k0*r1);

[ eH2, psiH2 ] = H2VibRotWaveFunction(R2, jRot, nVib);

eH2

P = LegendreP(jRot, Theta.x);
P = sqrt(jRot+1/2)*P;

psiP = psiH2*P';
psiP = reshape(psiP, [1, numel(psiP)]);

psi = zeros(2*n1, n2*nTheta);

psi(1:2:end, :) = real(g).'*psiP;
psi(2:2:end, :) = imag(g).'*psiP;

psi = reshape(psi, [2*n1, n2, nTheta]);

