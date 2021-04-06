function [U,varargout] = gt_cp(X,R,varargin)
% GT_CP Compute a CP decomposition using Genten
%
%  GT_CP makes available the CP decomposition methods in Genten to Matlab.
%  It works similarly to, and is mostly compatible with, the CP_ALS
%  function provided by the Tensor Toolbox (and therefore the Tensor
%  Toolbox must be in your Matlab path).  See CP_ALS for more information.
%  The differences with CP_ALS are:
%    - The input tensor must be a sparse tensor
%    - The 'dimorder' parameter is not accepted
%    - The 'init' option can only be used to specify an initial guess as a
%      ktensor (whereas CP_ALS accepts a variety of options).  If the intial
%      guess is not specified, a random initial guess is used.
%
%  Usage:
%
%  M = GT_CP(X,R) computes an estimate of the best rank-R
%   CP model of a sparse tensor X using an alternating least-squares
%   algorithm.  The result M is a ktensor.
%
%  [M,U0] = GT_CP(...) also returns the initial guess.
%
%  M = GT_CP(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'init' - Initial guess as a Ktensor (no default and uses random guess)
%      'printitn' - Print fit every n iterations; 0 for no printing {1}
%      'method' - CP method [{'cp-als'}, 'gcp']
%      'rol' - ROL options file name for GCP method {'rol.xml'}
%      'type' - Loss type for GCP method [{'gaussian'}, 'rayleigh', 'gamma',
%           'bernoulli', 'poisson']
%      'eps' - Ktensor entry shift for GCP method {1e-10}
%      'seed' - random number seed for random initial guess {12345}
%      'prng' - use parallel random number generator [{0}, 1]
%      'mttkrp_method' - method for handling parallel race conditions
%            in MTTKRP [{'atomic'}, 'duplicated', 'single', 'perm']
%      'mttkrp_tile_size' - number of factor matric columns to process
%            in each MTTKRP for the 'duplicated' method {0, which
%            means all columns}
%
%  See also CP_ALS, SPTENSOR_GT, KTENSOR

args = varargin;
if isa(X, 'sptensor_gt')
  a = X.alg_params; % For some reason, X.alg_params{:} only pulls out the first entry here?
  args = { a{:}, args{:} };
end
argout = cell(1,nargout-1);
[U,argout{:}] = gt_cp_driver(X,R,args{:});
varargout = argout;
