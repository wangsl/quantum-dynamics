
function [ psi, eHCl, psiHCl ] = InitWavePacket(R1, R2, Theta, jRot, nVib)

n1 = R1.n;
n2 = R2.n;
nTheta = Theta.n;

r1 = R1.r;
delta = R1.delta;
r10 = R1.r0;
k0 = R1.k0;

g = (1/(pi*delta^2))^(1/4) * ...
    exp(-(r1-r10).^2/(2*delta*delta) - j*k0*r1);

sum(conj(g).*g)*R1.dr

[ eHCl, psiHCl ] = HClVibRotWaveFunction(R2, jRot, nVib);

eHCl

sum(psiHCl.^2)*R2.dr

%HClWaveFunctionAnalysis(R2.r, psiHCl, R2.mass, jRot)

P = LegendreP(jRot, Theta.x);
P = sqrt(jRot+1/2)*P;

sum(P.^2.*Theta.w)

psiP = psiHCl*P';
psiP = reshape(psiP, [1, numel(psiP)]);

psi = zeros(2*n1, n2*nTheta);

psi(1:2:end, :) = real(g).'*psiP;
psi(2:2:end, :) = imag(g).'*psiP;

psi = reshape(psi, [2*n1, n2, nTheta]);


