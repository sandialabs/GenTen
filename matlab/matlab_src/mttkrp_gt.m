function v = mttkrp_gt(X,u,n,varargin)
% GT_MTTKRP Compute an MTTKRP using Genten
%
%  Usage:
%
%  V = MTTKRP_GT(X,U,N) computes an MTTKRP for a tensor X, ktensor U,
%  dimension N, and resulting factor matrix V.  Currently X must be a
%  sparse tensor in sptensor or sptensor_gt format.
%
%  See also SPTENSOR, SPTENSOR_GT, KTENSOR

v = gt_mttkrp_driver(X,u,n,varargin{:});
