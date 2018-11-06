function a = nnz(t)
%NNZ Number of nonzeros in sparse tensor.
%
%   NNZ(T) is the number of nonzero elements in T.
%
%   See also SPTENSOR_GT, SPTENSOR_GT/FIND.


if isempty(t.subs)
    a = 0;
else
    a = size(t.subs,2);
end
