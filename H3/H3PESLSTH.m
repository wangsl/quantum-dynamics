
% $Id$

% Reference
% J. Chem. Phys. 68, 2466 (1978)
% J. Chem. Phys. 71, 1514 (1979)

function [ V ] = H3PESLSTH(r1, r2, r3)

persistent first

if isempty(first)
  fprintf(1, 'To use LSTH PES\n')
  first = 0;
end

EH2Min = -1.1744746 + 1;

VI = H3VI(r1, r2, r3);

VII = H3VII(r1, r2, r3);

V = VI + VII - EH2Min;

return


% H2 potential energy curve

function [ pot ] = H2Pot(r)

persistent H2Spline

if length(find(r < 0.4))
  error('H2Pot r minimum is 0.4');
end

% Data from J. Chem. Phys. 43, 2429 (1965) Table I

R = [ 0.4000; 0.4500; 0.5000; 0.5500; 0.6000; 0.6500; 0.7000; 0.7500; ...
      0.8000; 0.9000; 1.0000; 1.1000; 1.2000; 1.3000; 1.3500; 1.3900; ...
      1.4000; 1.4010; 1.4011; 1.4100; 1.4500; 1.5000; 1.6000; 1.7000; ...
      1.8000; 1.9000; 2.0000; 2.1000; 2.2000; 2.3000; 2.4000; 2.5000; ...
      2.6000; 2.7000; 2.8000; 2.9000; 3.0000; 3.1000; 3.2000; 3.3000; ...
      3.4000; 3.5000; 3.6000; 3.7000; 3.8000; 3.9000; 4.0000; 4.1000; ...
      4.2000; 4.3000; 4.4000; 4.5000; 4.6000; 4.7000; 4.8000; 4.9000; ...
      5.0000; 5.1000; 5.2000; 5.3000; 5.4000; 5.5000; 5.6000; 5.7000; ...
      5.8000; 5.9000; 6.0000; 6.1000; 6.2000; 6.3000; 6.4000; 6.5000; ...
      6.6000; 6.7000; 6.8000; 6.9000; 7.0000; 7.2000; 7.4000; 7.6000; ...
      7.8000; 8.0000; 8.2500; 8.5000; 9.0000; 9.5000; 10.0000;
    ];
      
V = [ -0.1202028; -0.3509282; -0.5266270; -0.6627707; -0.7696341; ...
      -0.8543614; -0.9220261; -0.9763357; -1.0200556; -1.0836442; ...
      -1.1245385; -1.1500562; -1.1649342; -1.1723459; -1.1739627; ...
      -1.1744517; -1.1744744; -1.1744746; -1.1744746; -1.1744599; ...
      -1.1740558; -1.1728537; -1.1685799; -1.1624570; -1.1550670; ...
      -1.1468496; -1.1381312; -1.1291562; -1.1201233; -1.1111725; ...
      -1.1024127; -1.0939273; -1.0857810; -1.0780164; -1.0706700; ...
      -1.0637641; -1.0573118; -1.0513185; -1.0457832; -1.0407003; ...
      -1.0360578; -1.0318402; -1.0280272; -1.0245978; -1.0215297; ...
      -1.0187967; -1.0163689; -1.0142247; -1.0123371; -1.0106810; ...
      -1.0092303; -1.0079682; -1.0068703; -1.0059178; -1.0050923; ...
      -1.0043782; -1.0037626; -1.0032309; -1.0027740; -1.0023800; ...
      -1.0020423; -1.0017521; -1.0015030; -1.0012899; -1.0011069; ...
      -1.0009498; -1.0008150; -1.0007002; -1.0006030; -1.0005162; ...
      -1.0004466; -1.0003864; -1.0003328; -1.0002906; -1.0002466; ...
      -1.0002154; -1.0001889; -1.0001434; -1.0001086; -1.0000868; ...
      -1.0000682; -1.0000528; -1.0000404; -1.0000314; -1.0000185; ...
      -1.0000121; -1.0000091;
    ];

if isempty(H2Spline)
  H2Spline = spline(R, [-5.3065581; V; 0.0000059]);
end

% Data from J. Chem. Phys. 68, 2466 (1978) Table I
% J. Chem. Phys. 71, 1514 (1979)

C6 = 6.899992032;
C8 = 219.9997304;

IndexLt10 = find(r <= 10.0);
IndexGt10 = find(r > 10.0);

pot = zeros(size(r));

pot(IndexLt10) = ppval(H2Spline, r(IndexLt10)) + 1;

r1 = r(IndexGt10);
r2 = r1.*r1;
r4 = r2.*r2;
r6 = r2.*r4;
r8 = r4.*r4;
pot(IndexGt10) = -C6./r6 - C8./r8;

return

% H2 triplet curve

function [ f ] = H2E3(r)

alpha1 = -1.2148730613;
alpha2 = -1.5146634740;
alpha3 = -1.46;
alpha4 = 2.088442;

f = alpha1*(alpha2 + r + alpha3*r.^2).*exp(-alpha4*r);

return

% 

function [ VLondon ] = H3VLondon(r1, r2, r3)

VH1 = H2Pot(r1);
VH2 = H2Pot(r2);
VH3 = H2Pot(r3);

E31 = H2E3(r1);
E32 = H2E3(r2);
E33 = H2E3(r3);

Q1 = 0.5*(VH1 + E31);
Q2 = 0.5*(VH2 + E32);
Q3 = 0.5*(VH3 + E33);

J1 = 0.5*(VH1 - E31);
J2 = 0.5*(VH2 - E32);
J3 = 0.5*(VH3 - E33);

VLondon = Q1 + Q2 + Q3 ...
	  - sqrt(0.5*((J2-J1).^2 + (J3-J2).^2 + (J3-J1).^2));

return

%

function [ Va ] = H3Va(r1, r2, r3)

a2 = 0.0012646477;
a3 = -0.0001585792;
a4 = 0.0000079707;
a5 = -0.0000001151;

alpha5 = 0.0035;

A = abs((r1-r2).*(r2-r3).*(r3-r1));
R = r1+r2+r3;

A2 = A.*A;
A3 = A2.*A;
A4 = A2.*A2;
A5 = A2.*A3;

Va = (a2*A2 + a3*A3 + a4*A4 + a5*A5).*exp(-alpha5*R.^3);

return

%

function [ VI ] = H3VI(r1, r2, r3)
VI = H3VLondon(r1, r2, r3) + H3Va(r1, r2, r3);
return

function [ VII ] = H3VII(r1, r2, r3)

beta1=0.52;
beta2=0.052;
beta3=0.79;
b11=3.0231771503;
b12=-1.0893521900;
b22=1.7732141742;
b23=-2.0979468223;
b24=-3.978850217;
b31=0.4908116374;
b32=-0.8718696387;
b41=0.1612118092;
b42=-0.1273731045;
b51=-13.3599568553;
b52=0.9877930913;

r1Sq = r1.*r1;
r2Sq = r2.*r2;
r3Sq = r3.*r3;

B1 = 1 ...
     + (r1Sq-r2Sq-r3Sq)./(2*r2.*r3) ...
     + (r2Sq-r1Sq-r3Sq)./(2*r1.*r3) ...
     + (r3Sq-r1Sq-r2Sq)./(2*r1.*r2);

B2 = 1./r1 + 1./r2 + 1./r3;

B3 = (r1-r2).^2 + (r2-r3).^2 + (r3-r1).^2;

R = r1+r2+r3;

B1_2 = B1.*B1;
B1_3 = B1.*B1_2;
B1_4 = B1_2.*B1_2;
R_2 = R.*R;

ExpBeta1R1 = exp(-beta1*R);
ExpBeta2R2 = exp(-beta2*R_2);

Vb1 = B1.*(b11+b12*R).*ExpBeta1R1;
Vb2 = (b22*B1_2 + b23*B1_3 + b24*B1_4).*ExpBeta2R2;
Vb3 = B2.*(b31.*B1.*ExpBeta1R1 + b32*B1_2.*ExpBeta2R2);
Vb4 = B3.*B1.*(b41*ExpBeta1R1 + b42*ExpBeta2R2);
Vb5 = B1.*(b51+b52*R_2).*exp(-beta3*R);

VII = Vb1 + Vb2 + Vb3 + Vb4 + Vb5;

return
