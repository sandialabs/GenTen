%SPTENSOR_GT Class for sparse tensors in Genten format.
%   See also TENSOR_TOOLBOX

function t = sptensor_gt(varargin)
%SPTENSOR_GT Create a sparse tensor.
%
%   X = SPTENSOR_GT(SUBS, VALS, SZ, FUN) uses the rows of SUBS and VALS
%   to generate a sparse tensor X of size SZ = [m1 m2 ... mn]. SUBS is
%   an p x n array specifying the subscripts of the values to be
%   inserted into S. The k-th row of SUBS specifies the subscripts for
%   the k-th value in VALS. The values are accumulated at repeated
%   subscripts using the function FUN, which is specified by a
%   function handle.
%
%   There are several simplifications of this four argument call.
%
%   X = SPTENSOR_GT(SUBS,VALS,SZ) uses FUN=@SUM.
%
%   X = SPTENSOR_GT(SUBS,VALS) uses SZ = max(SUBS,[],1).
%
%   X = SPTENSOR_GT(SZ) abbreviates X = SPTENSOR_GT([],[],SZ).
%
%   X = SPTENSOR_GT(Y) copies/converts Y if it is an sptensor_gt, sptensor,
%   an sptenmat, or
%   a dense tensor or MDA (the zeros are squeezed out), an sptensor3, or a
%   sparse matrix. Note that a row-vector, integer MDA is interpreted as a
%   size (see previous constructor).
%
%   X = SPTENSOR_GT is the empty constructor.
%
%   X = SPTENSOR_GT(FH,SZ,NZ) creates a random sparse tensor of the specified
%   size with NZ nonzeros (this can be an explit value or a proportion).
%   The function handle FH is used to create the nonzeros.
%
%   The argument VALS may be scalar, which is expanded to be the
%   same length as SUBS, i.e., it is equivalent to VALS*(p,1).
%
%   Examples
%   subs = [1 1 1; 1 1 3; 2 2 2; 4 4 4; 1 1 1; 1 1 1]
%   vals = [0.5; 1.5; 2.5; 3.5; 4.5; 5.5]
%   siz = [4 4 4];
%   X = sptensor_gt(subs,vals,siz) %<-- sparse 4x4x4, repeats summed
%   X = sptensor_gt(subs,1,siz) %<-- scalar 2nd argument
%   X = sptensor_gt(subs,vals,siz,@max) %<-- max for accumulation
%   myfun = @(x) sum(x) / 3;
%   X = sptensor_gt(subs,vals,siz,myfun) %<-- custom accumulation
%
%   See also SPTENSOR_GT, SPTENSOR.
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


% EMPTY Constructor
if (nargin == 0) || ((nargin == 1) && isempty(varargin{1}))
    t.subs = [];
    t.vals = [];
    t.size = [];
    t.perm = [];
    t.alg_params = {};
    t = class(t,'sptensor_gt');
    return;
end

% SINGLE ARGUMENT
if (nargin == 1)

    source = varargin{1};

    switch(class(source))

        % COPY CONSTRUCTOR
        case 'sptensor_gt',
            t.subs = source.subs;
            t.vals = source.vals;
            t.size = source.size;
            t.perm = source.perm;
            t.alg_params = source.alg_params;
            t = class(t, 'sptensor_gt');
            return;

        % CONVERT SPTENSOR
        case 'sptensor',
            t.subs = uint64(source.subs-1)';
            t.vals = source.vals;
            t.size = uint64(source.size);
            t.perm = [];
            t.alg_params = {};
            t = class(t, 'sptensor_gt');
            return;

        % CONVERT ANYTHING SUPPORTED BY SPTENSOR
        otherwise,
            t = sptensor_gt(sptensor(source));
            return;

    end % switch

end % nargin == 1

% SPECIAL CASE for INTERACTION WITH MEX FILES OR DIRECT CREATION OF
% SPTENSOR_GT WITHOUT ANY SORTING OR OTHER STANDARD CHECKS
if (nargin == 4) && (isnumeric(varargin{4})) && (varargin{4} == 0)

    % Store everything
    t.subs = varargin{1};
    t.vals = varargin{2};
    t.size = varargin{3};
    t.perm = [];
    t.alg_params = {};

    % Create the tensor
    t = class(t, 'sptensor_gt');

    return;

end

% ANYTHING ELSE
t = sptensor_gt(sptensor(varargin{:}));
return;
