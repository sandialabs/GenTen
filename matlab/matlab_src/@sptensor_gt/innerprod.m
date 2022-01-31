function res = innerprod(X,Y,varargin)
%INNERPROD Efficient inner product with a sparse tensor.
%
%   R = INNERPROD(X,Y) efficiently computes the inner product between
%   two tensors X and Y.  If Y is a tensor or sptensor, the inner
%   product is computed directly and the computational complexity is
%   O(min(nnz(X),nnz(Y))). If Y is a ktensor or a ttensor, the
%   inner product method for that type of tensor is called.
%
%   See also SPTENSOR_GT, KTENSOR/INNERPROD, TTENSOR/INNERPROD.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt


% X is a sptensor
if nnz(X) == 0 %There are no nonzero terms in X.
    res = 0;
    return
end

switch class(Y)

    case {'sptensor','sptensor_gt'}
        if ~isequal(size(X),size(Y))
            error('X and Y must be the same size.');
        end
        if nnz(Y) == 0 %There are no nonzero terms in Y.
            res = 0;
            return
        end
        if nnz(X) < nnz(Y);
            [SX,VX] = find(X);
            VY = extract(Y,SX);   %<-----VY = Y(SX);
        else
            [SY,VY] = find(Y);
            VX = extract(X,SY);   %<-----VX = X(SY);
        end
        res = VY'*VX;
        return;

    case {'tensor'}
        if ~isequal(size(X),size(Y))
            error('X and Y must be the same size.');
        end
        [SX,VX] = find(X);
        VY = extract(Y,SX);
        res = VY'*VX;
        return;

    case {'ktensor'}
        % Call MEX driver
        args = { X.alg_params{:}, varargin{:} };
        res = gt_innerprod_driver(X,Y,args{:});
        return;

    otherwise
        error(['Inner product not available for class ' class(Y)]);

end
