
function [ V ] = OOPES2(rOO)

r1(1) = rOO;
r3(1) = 120.0;

r2 = r1+r3;

V = DMBEIVMex(r1, r2, r3);
