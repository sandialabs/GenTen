function [U,varargout] = cp_als(X,R,varargin)
%CP_ALS Compute a CP decomposition using GenTen
%
%  This is just a wrapper for CP_ALS_GT allowing CP_ALS to be called for tensors
%  in the SPTENSOR_GT format, without using a new name.  See those functions for
%  documentation.
%
%  See also CP_ALS_GT, SPTENSOR/CP_ALS, SPTENSOR_GT, KTENSOR

argout = cell(1,nargout-1);
[U,argout{:}] = cp_als_gt(X,R,varargin{:});
varargout = argout;