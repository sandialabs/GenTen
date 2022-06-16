function [U,varargout] = cp_als(X,R,varargin)
%CP_ALS Compute a CP decomposition using GenTen
%
%  This is just a wrapper for CP_ALS_GT allowing CP_ALS to be called for tensors
%  in the SPTENSOR_GT format, without using a new name.  See those functions for
%  documentation.
%
%  See also CP_ALS_GT, SPTENSOR/CP_ALS, SPTENSOR_GT, KTENSOR

% Check for argument struct instead of list
argin = varargin;
if length(argin) == 1 && isa(argin{1}, 'struct')
  arg_names = fieldnames(argin{1});
  arg_vals = struct2cell(argin{1});
  argin = reshape({arg_names{:}; arg_vals{:}}, [2*length(arg_names),1]);
end

% Add alg params from X tensor
% Actually this is done in cp_als_gt now
%argin = { X.alg_params{:}, argin{:} };

argout = cell(1,nargout-1);
[U,argout{:}] = cp_als_gt(X,R,argin{:});
varargout = argout;
