
% $Id$

function [ V ] = H3PES(r1, r2, r3)

global UseLSTH

if UseLSTH 
  V = H3PESLSTH(r1, r2, r3);
else
  V = H3PESBKMP2(r1, r2, r3);
end
