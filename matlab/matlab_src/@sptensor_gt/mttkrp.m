function v = mttkrp(X,u,n,varargin)
% MTTKRP Compute an MTTKRP using Genten
%
%  Usage:
%
%  V = MTTKRP(X,U,N) computes an MTTKRP for a tensor X, ktensor U,
%  dimension N, and resulting factor matrix V.  Currently X must be a
%  sparse tensor in sptensor_gt format.
%
%  See also SPTENSOR_GT, SPTENSOR/MTTKRP, KTENSOR

args = { X.alg_params{:}, varargin{:} };
v = gt_mttkrp_driver(X,u,n,args{:});
