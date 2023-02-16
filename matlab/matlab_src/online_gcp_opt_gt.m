function [u, info] = online_gcp_opt(X, nc, slice_size, printitn, temporal_solver_params, spatial_solver_params, varargin)
%ONLINE_GCP_OPT Fits Online Generalized CP decomposition with user-specified function.

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

args = { 'printitn', printitn, varargin{:} };

% Compute initial guess
[init,args] = gt_get_option_and_remove(args,'init','rand');
if iscell(init)
  u0 = ktensor(init);
elseif isa(init, 'ktensor')
  u0 = init;
elseif strcmp(init,'rand') || strcmp(init,'randn')
  uinit = cell(nd,1);
  for k=1:(nd-1)
    if strcmp(init,'rand')
      uinit{k} = rand(sz(k),nc);
    else
      uinit{k} = randn(sz(k),nc);
    end
    u0 = ktensor(uinit);
  end
end

% Add tensor alg_params
if length(XX) >= 1 && isa(XX{1}, 'sptensor_gt')
  a = XX{1}.alg_params; % XX{1}.alg_params{:} only extracts the first entry?
  args = [ a args ];
end

% Parse loss type
[type,args] = gt_get_option_and_remove(args,'type','normal');
if strcmp(type,'normal')
  type = 'gaussian';
elseif strcmp(type,'count')
  type = 'poisson';
elseif strcmp(type,'binary')
  type = 'bernoulli';
end
c = {'type', type};
args = [args c];
temporal_solver_params.type = type;
spatial_solver_params.type = type;

% Xinit
[Xinit,args] = gt_get_option_and_remove(args,'Xinit',[]);
if isempty(Xinit)
  Xinit = sptensor_gt;
end
if ~isa(Xinit, 'sptensor') && ~isa(Xinit, 'sptensor_gt')
  error('Xinit must be an sptensor or sptensor_gt');
end

% Streaming solver
[use_time_lss_for_gaussian,args] = gt_get_option_and_remove(args,'use_time_lss_for_gaussian',0);
[use_space_lss_for_gaussian,args] = gt_get_option_and_remove(args,'use_space_lss_for_gaussian',0);
[use_ocp_for_gaussian,args] = gt_get_option_and_remove(args,'use_ocp_for_gaussian',0);
if use_time_lss_for_gaussian || use_ocp_for_gaussian
  temporal_solver_params.streaming_solver = 'least-squares';
end
if use_space_lss_for_gaussian
  spatial_solver_params.streaming_solver = 'least-squares';
end
if use_ocp_for_gaussian
  spatial_solver_params.streaming_solver = 'online-cp';
end

% Parse options with just a different name
args = gt_rename_option(args,'window_size','window-size');
args = gt_rename_option(args,'window_method','window-method');
args = gt_rename_option(args,'window_weight','window-weight');
args = gt_rename_option(args,'window_penalty','window-penalty');
args = gt_rename_option(args,'window_term','history-method');
args = gt_rename_option(args,'penalty','factor-penalty');

% Unsupported options
[func,args] = gt_get_option_and_remove(args,'func',[]);
if ~isempty(func)
  error('Option \''func\'' is not supported');
end
[grad,args] = gt_get_option_and_remove(args,'grad',[]);
if ~isempty(grad)
  error('Option \''grad\'' is not supported');
end
[lower,args] = gt_get_option_and_remove(args,'lower',[]);
if ~isempty(lower)
  error('Option \''lower\'' is not supported');
end
[state,args] = gt_get_option_and_remove(args,'state',[]);
if ~isempty(state)
  error('Option \''state\'' is not supported');
end
[normalize,args] = gt_get_option_and_remove(args,'normalize',0);
if normalize
  error('Option \''normalize\'' is not supported');
end
[window_weight_method,args] = gt_get_option_and_remove(args,'window_weight_method','exp');
if strcmp(window_weight_method,'const')
  [window_weight,args] = gt_get_option_and_remove(args,'window_weight',1);
  if window_weight ~= 1
    error('Option \''window_weight_method\'' is not supported');
  end
elseif ~strcmp(window_weight_method,'exp')
  error('Option \''window_weight_method\'' is not supported');
end
[include_init_in_window,args] = gt_get_option_and_remove(args,'include_init_in_window',1);
if ~include_init_in_window
  error('Option \''include_init_in_window\'' is not supported');
end
[monitor_optimization,args] = gt_get_option_and_remove(args,'monitor_optimization',false);
if monitor_optimization
  error('Option \''monitor_optimization\'' is not supported');
end
[include_his_in_F,args] = gt_get_option_and_remove(args,'include_his_in_F',1);
if ~include_his_in_F
  error('Option \''include_his_in_F\'' is not supported');
end
[include_reg_in_F,args] = gt_get_option_and_remove(args,'include_reg_in_F',1);
if ~include_reg_in_F
  error('Option \''include_reg_in_F\'' is not supported');
end

% Translate temporal/spatial solver params
if isfield(temporal_solver_params,'sampler')
  if strcmp(temporal_solver_params.sampler, 'stratified-gt')
    temporal_solver_params.sampler = 'stratified';
  end
  if strcmp(temporal_solver_params.sampler, 'semi-stratified-gt')
    temporal_solver_params.sampler = 'semi-stratified';
  end
end
if isfield(spatial_solver_params,'sampler')
  if strcmp(spatial_solver_params.sampler, 'stratified-gt')
    spatial_solver_params.sampler = 'stratified';
  end
  if strcmp(spatial_solver_params.sampler, 'semi-stratified-gt')
    spatial_solver_params.sampler = 'semi-stratified';
  end
end
temporal_solver_params = gt_translate_gcp_options(temporal_solver_params);
spatial_solver_params = gt_translate_gcp_options(spatial_solver_params);

% Call genten
[u,fest,ften] = gt_online_gcp(XX, u0, temporal_solver_params, spatial_solver_params, Xinit, args{:});

info.fest = fest;
info.ften = ften;

mainTime = toc(mainStart);

fprintf('\nMain loop time: %.2f seconds\n', mainTime);

end
