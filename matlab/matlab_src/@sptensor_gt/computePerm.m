function t = computePerm(t)
%COMPUTEPERM Compute permutation array for sparse tensor used in MTTKRP
%
%   COMPUTEPERM(T) returns a new tensor storing a permutation array used in
%   MTTKRP.  Use this to prevent the permutation array from being recreated
%   every type GT_CP() is called.
%
%   See also GT_CP, SPTENSOR_GT.
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

t.perm = gt_compute_perm_driver(t);
