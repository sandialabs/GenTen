function [u, info] = online_gcp_opt(X, nc, slice_size, printitn, temporal_solver_params, spatial_solver_params, varargin)
%ONLINE_GCP_OPT Fits Online Generalized CP decomposition with user-specified function.

parser = inputParser;
parser.KeepUnmatched = true;
parser.addParameter('init','rand');
parser.addParameter('Xinit',[]);
parser.parse(varargin{:});

mainStart = tic;

% Parse tensor slices
if isa(X,'cell')  % slices are in a cell array
  num_time = length(X);
  sz = size(X{1});
  nd = length(sz);
  XX = X;
else              % slices are in one tensor
  sz = size(X);
  nd = length(sz);
  num_time = sz(nd);
  sz = [sz(1:nd-1) slice_size];
  num_slices = ceil(num_time/slice_size);
  XX = cell(num_slices,1);
  for i=1:num_slices
    % Size of our slice
    if slice < num_slices
      ssz = slice_size;
    else
      ssz = num_time - (num_slices-1)*slice_size;
    end
    ind = (slice-1)*slice_size + [1:ssz];

    % Extract tensor slice
    S.subs = repmat({':'},1,nd);
    S.subs{nd} = ind;
    S.type = '()';
    XX{i} = reshape(subsref(X, S), [sz(1:nd-1) ssz]);
  end
end

% Compute initial guess if this is the first slice
if iscell(parser.Results.init)
  u0 = ktensor(parser.Results.init);
elseif isa(parser.Results.init, 'ktensor')
  u0 = parser.Results.init;
elseif strcmp(parser.Results.init,'rand') || strcmp(parser.Results.init,'randn')
  uinit = cell(nd,1);
  for k=1:(nd-1)
    if strcmp(parser.Results.init,'rand')
      uinit{k} = rand(sz(k),nc);
    else
      uinit{k} = randn(sz(k),nc);
    end
    u0 = ktensor(uinit);
  end
end

% Setup args
arg_names = fieldnames(parser.Unmatched);
arg_vals = struct2cell(parser.Unmatched);
args = reshape({arg_names{:}; arg_vals{:}}, [2*length(arg_names),1]);
if length(XX) >= 1 && isa(XX{1}, 'sptensor_gt')
  a = XX{1}.alg_params; % XX{1}.alg_params{:} only extracts the first entry?
  args = { a{:}, args{:} };
end

Xinit = parser.Results.Xinit;
if isempty(Xinit)
  Xinit = sptensor_gt;
end
if ~isa(Xinit, 'sptensor') && ~isa(Xinit, 'sptensor_gt')
  error('Xinit must be an sptensor or sptensor_gt');
end

% Call genten
[u,fest,ften] = gt_online_gcp(XX, u0, temporal_solver_params, spatial_solver_params, Xinit, args{:});

info.fest = fest;
info.ften = ften;

mainTime = toc(mainStart);

fprintf('\nMain loop time: %.2f seconds\n', mainTime);

end
