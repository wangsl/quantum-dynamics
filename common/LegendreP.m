
% $Id$

function [ v ] = LegendreP(n, x)

if n == 0
  v = ones(size(x));
  return
elseif n == 1
  v = x;
  return
end

p0 = 1;
p1 = x;

for i = 2 : n
  p2 = (2-1/i)*x.*p1 - (1-1/i).*p0;
  p0 = p1;
  p1 = p2;
end

v = p2;
