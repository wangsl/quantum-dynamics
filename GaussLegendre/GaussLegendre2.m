function [ x, w ] = GaussLegendre2(m)

assert( 1 <= m && m <= 599)

data_dir = '/home/wang/matlab/quantum-dynamics/GaussLegendre';

grids_file = strcat(data_dir, '/', int2str(m), '-LegendreGauss.grids');

if exist(grids_file, 'file') ~= 2 
  error('"%s" does not exit\n', grids_file)
end

[ x, w ] = textread(grids_file, '%f %f');

assert(isequal(size(x), size(w)))
assert(m == 2*numel(x) - mod(m,2))

if mod(m, 2) 
  x = cat(1, -x(end:-1:2), x);
  w = cat(1,  w(end:-1:2), w);
else
  x = cat(1, -x(end:-1:1), x);
  w = cat(1,  w(end:-1:1), w);
end

return
