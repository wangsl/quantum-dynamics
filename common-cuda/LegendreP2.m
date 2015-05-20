
% $Id$

function [ v ] = LegendreP2(n, x)

x = reshape(x, [numel(x), 1]);
v = zeros(numel(x), n+1);

if n == 0
  v(:,1) = 1;
  return
elseif n == 1
  v(:,1) = 1;
  v(:,2) = x;
  return
end

v(:,1)= 1;
v(:,2)= x;

for i = 2 : n
  v(:,i+1) = (2-1/i)*x.*v(:,i) - (1-1/i)*v(:,i-1);
end

