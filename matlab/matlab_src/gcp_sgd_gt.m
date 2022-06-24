function [U,varargout] = gcp_sgd_gt(X,R,varargin)
%GCP_SGD_GT Compute a Generalized CP decomposition using GenTen
%
%  GCP_SGD_GT makes the GCP decomposition method implementation in GenTen
%  available to Matlab.  The input and output arguments are intented to be
%  as compatible with the GCP_OPT function in the Tensor Toolbox as possible,
%  and relies on the Tensor Toolbox for a lot of its functionality (and
%  therefore the Tensor Toolbox must be in your Matlab path).  See GCP_OPT
%  for more information.
%
%  The differences with GCP_OPT are:
%    - The input tensor must be a sparse tensor as represented by the
%      the sptensor or sptensor_gt class.  The latter is the native format in
%      GenTen and hence is preferred (if the input tensor is in sptensor format
%      it will be copied to sptensor_gt format internally).
%    - Only the 'sgd' optimization method is supported
%    - Only the 'stratified' f-sampler is supported along with 'stratified' and
%      'semi-stratified' g-sampler
%    - The 'func', 'grad', 'lower', 'mask', 'state', 'factr', and 'pgtol'
%      options are not supported
%    - In addition to the options for specifying the initial guess 'init'
%      supported by GCP_OPT, GCP_SGD_GT also supports the intial guess as a
%      ktensor as well as 'random_gt', for a random initial guess generated
%      by GenTen.
%    - The initial guess is always returned as a ktensor instead of a cell array
%    - Numerous other options are supported, add 'help' as an option to see a
%      list
%
%  See also GCP_OPT, SPTENSOR_GT, KTENSOR

%
% Parse arguments
%
args = varargin;
if isa(X, 'sptensor_gt')
  a = X.alg_params; % For some reason, X.alg_params{:} only pulls out the first entry here?
  args = { a{:}, args{:} };
end
N = ndims(X);

% Initial guess
Uinit = 'random';
n = length(args);
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i}, 'init')
    if i+1<=n && isa(args{i+1}, 'char') && strcmp(args{i+1}, 'random')
      Uinit = 'random';
      args(i:i+1) = [];
      break;
    elseif i+1<=n && isa(args{i+1}, 'char') && strcmp(args{i+1}, 'nvecs')
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
if isa(Uinit, 'char') && strcmp(Uinit, 'random')
  Uinit = cell(N,1);
  for j = 1:N
    Uinit{j} = rand(size(X,j),R);
  end
  Uinit = ktensor(Uinit);
end

% translate options
[type,args] = get_option_and_remove(args,'type','normal');
if strcmp(type,'normal')
  type = 'gaussian';
elseif strcmp(type,'count')
  type = 'poisson';
elseif strcmp(type,'binary')
  type = 'bernoulli';
end
c = {'type', type};
args = [args c];
[fsamp,args] = get_option_and_remove(args,'fsamp',[]);
if ~isempty(fsamp)
  if length(fsamp) == 1
    f = fsamp(1);
    if f == Inf
      f = nnz(X);
    end
    c = {'fnzs', f, 'fzs', f};
  else
    f1 = fsamp(1);
    f2 = fsamp(2);
    if f1 == Inf
      f1 = nnz(X);
    end
    if f2 == Inf
      f2 = prod(size(X)) - nnz(X);
    end
    c = {'fnzs', f1, 'fzs', f2};
  end
  args = [args c];
end
[gsamp,args] = get_option_and_remove(args,'gsamp',[]);
if ~isempty(gsamp)
  if length(gsamp) == 1
    g = gsamp(1);
    if g == Inf
      g = nnz(X);
    end
    c = {'gnzs', g, 'gzs', g};
  else
    g1 = gsamp(1);
    g2 = gsamp(2);
    if g1 == Inf
      g1 = nnz(X);
    end
    if g2 == Inf
      g2 = prod(size(X)) - nnz(X);
    end
    c = {'gnzs', g1, 'gzs', g2};
  end
  args = [args c];
end
args = rename_option(args,'sampler','sampling');
args = rename_option(args,'maxfails','fails');
args = rename_option(args,'epciters','epochiters');
args = rename_option(args,'festtol','gcp-tol');
args = rename_option(args,'beta1','adam-beta1');
args = rename_option(args,'beta2','adam-beta2');
args = rename_option(args,'epsilon','adam-eps');

% Call driver
argout = cell(1,nargout-1);
[U,argout{:}] = gt_cp_driver(X,R,Uinit,'method','gcp-sgd',args{:});
varargout = argout;

% Find an option with the given name and remove it from the list
function [v,args] = get_option_and_remove(args,name,dflt)
n = length(args);
v = dflt;
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i},name)
    v = args{i+1};
    args(i:i+1) = [];
    break;
  end
end

% Rename an option
function args = rename_option(args,name,new_name)
n = length(args);
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i},name)
    args{i} = new_name;
    break;
  end
end
