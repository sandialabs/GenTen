function [U,varargout] = cp_opt_gt(X,R,varargin)
%CP_OPT_GT Compute a CP decomposition using GenTen
%
%  CP_OPT_GT makes the CP-OPT decomposition method implementation in GenTen
%  available to Matlab.  The input and output arguments are intented to be
%  as compatible with the CP_OPT function in the Tensor Toolbox as possible,
%  and relies on the Tensor Toolbox for a lot of its functionality (and
%  therefore the Tensor Toolbox must be in your Matlab path).  See CP_OPT
%  for more information.
%
%  The differences with CP_OPT are:
%    - The input tensor must be a sparse tensor as represented by the
%      the sptensor or sptensor_gt class.  The latter is the native format in
%      GenTen and hence is preferred (if the input tensor is in sptensor format
%      it will be copied to sptensor_gt format internally).
%    - The 'dimorder' parameter is not accepted.
%    - In addition to the options for specifying the initial guess 'init'
%      supported by CP_OPT, CP_OPT_GT also supports the intial guess as a
%      ktensor as well as 'random_gt', for a random initial guess generated
%      by GenTen.
%    - The initial guess is always returned as a ktensor instead of a cell array
%
%  CP_OPT_GT accepts the following additional options:
%      'seed' - random number seed for random initial guess {12345}
%      'prng','no-prng' - use parallel random number generator (default: no)
%      'timings','no-timings' - print accurate kernel timing info (but may
%            increase total run time by adding fences) (default: no)
%      'mttkrp-all-method' - method for handling parallel race conditions
%            in MTTKRP ['atomic', 'duplicated', 'single', 'iterated']
%            (default depends on the architectrure).
%      'mttkrp-nnz-tile-size'  Nonzero tile size for mttkrp algorithm [{128}]
%      'mttkrp-duplicated-tile-size' Factor matrix tile size for duplicated
%            mttkrp algorithm (0 means all columns) [{0}]
%      'mttkrp-duplicated-threshold' Theshold for determining when to not
%            use duplicated mttkrp algorithm (set to -1.0 to always use
%            duplicated) [{-1.0}]
%
%  See also CP_OPT, SPTENSOR_GT, KTENSOR

%
% Parse arguments
%
args = varargin;
N = ndims(X);

% Check for argument struct instead of list
if length(args) == 1 && isa(args{1}, 'struct')
  arg_names = fieldnames(args{1});
  arg_vals = struct2cell(args{1});
  args = reshape({arg_names{:}; arg_vals{:}}, [2*length(arg_names),1]);
end

% Initial guess
Uinit = 'randn';
n = length(args);
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i}, 'init')
    if i+1<=n && isa(args{i+1}, 'char') && (strcmpi(args{i+1}, 'random') || strcmpi(args{i+1}, 'randn') || strcmpi(args{i+1}, 'rand') || strcmpi(args{i+1}, 'zeros'))
      Uinit = args{i+1};
      args(i:i+1) = [];
      break;
    elseif i+1<=n && isa(args{i+1}, 'char') && strcmpi(args{i+1}, 'nvecs')
      Uinit = cell(N,1);
      for j = 1:N
        Uinit{j} = nvecs(X,j,R);
      end
      Uinit = ktensor(Uinit);
      args(i:i+1) = [];
      break;
    elseif i+1<=n && isa(args{i+1}, 'cell')
      Uinit = ktensor(args{i+1}); % convert cell array to ktensor
      args(i:i+1) = [];
      break;
    elseif i+1<=n && isa(args{i+1}, 'ktensor')
      Uinit = args{i+1};
      args(i:i+1) = [];
      break;
    elseif i+1<=n && isa(args{i+1}, 'char') && strcmp(args{i+1}, 'random_gt')
      Uinit = 'random_gt';
      args(i:i+1) = [];
      break;
    end
  end
end
if isa(Uinit, 'char') && (strcmpi(Uinit, 'random') || strcmpi(Uinit, 'randn') || strcmpi(Uinit, 'rand') || strcmpi(Uinit, 'zeros'))
  init = Uinit;
  Uinit = cell(N,1);
  for j = 1:N
    Uinit{j} = matrandnorm(feval(init,size(X,j),R));
  end
  Uinit = ktensor(Uinit);
end

% Option parsing
fs = 1;
n = length(args);
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i}, 'fixsigns')
    if i+1<=n && (isa(args{i+1}, 'logical') || isa(args{i+1}, 'double'))
      fs = args{i+1};
      args(i:i+1) = [];
      break;
    end
  end
end

% Call driver
argout = cell(1,nargout-1);
[U,argout{:}] = gt_cp_driver(X,R,Uinit,'method','cp-opt',args{:});
varargout = argout;

% Fix signs
if fs
  U = fixsigns(U);
end
