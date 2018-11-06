function t = sptensor(X)
%SPTENSOR Convert a sparse tensor in sptensor_gt format to sptensor format.
%
%   See also SPTENSOR_GT.

t = sptensor(double(X.subs)'+1, X.vals, X.size);
return;
