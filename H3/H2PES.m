
% $Id$

function [ V ] = H2PES(r)

R = zeros(size(r));
R(:) = 100.0;

V = H3PES(r, R, r+R);

return
