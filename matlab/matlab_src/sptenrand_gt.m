function Y = sptenrand_gt(sz,nz)
%SPTENRAND_GT Sparse uniformly distributed random tensor in Genten format.
%
%   R = SPTENRAND_GT(sz,density) creates a random sparse tensor of the
%   specified sz with approximately density*prod(sz) nonzero
%   entries.
%
%   R = SPTENRAND_GT(sz,nz) creates a random sparse tensor of the
%   specified sz with approximately nz nonzero entries.
%
%   Example: R = sptenrand_gt([5 4 2],12);
%
%   See also SPTENSOR_GT, RAND.
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.



% Error check on siz
if ndims(sz) ~= 2 || size(sz,1) ~= 1
    error('Size must be a row vector');
end

% Error check on nz
if ~exist('nz','var') || (nz < 0)
    error('2nd argument must be positive');
end

% Is nz an number or a fraction? Ultimately want a number.
if (nz < 1)
    nz = prod(sz) * nz;
end

% Make sure nz is an integer
nz = ceil(nz);

% Keep iterating until we find enough unique nonzeros or we
% give up
subs = [];
cnt = 0;
while (size(subs,1) < nz) && (cnt < 10)
    subs = ceil( rand(nz, size(sz,2)) * diag(sz) );
    subs = unique(subs, 'rows');
    cnt = cnt + 1;
end

% Extract nnz subscipts and create a corresponding list of
% values
nz = min(nz, size(subs,1));
subs = subs(1:nz,:);
vals = rand(nz,1);

Y = sptensor_gt(subs,vals,sz);
return;
