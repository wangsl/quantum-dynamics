
function [ x, w ] = GaussLegendre2(n)

if n < 1 || n > 599 
  error('argument should be 1 <= n <= 599')
end

data_dir = '/home/wang/matlab/quantum-dynamics/GaussLegendre';

grids_file = strcat(data_dir, '/', int2str(n), '-LegendreGauss.grids');

if exist(grids_file, 'file') ~= 2 
  error('"%s" does not exit\n', grids_file)
end

[ x, w ] = textread(grids_file, '%f %f');

assert(isequal(size(x), size(w)))
assert(n == 2*numel(x) - mod(n,2))

if mod(n, 2) 
  x = cat(1, -x(end:-1:2), x);
  w = cat(1,  w(end:-1:2), w);
else
  x = cat(1, -x(end:-1:1), x);
  w = cat(1,  w(end:-1:1), w);
end

return
