function args = gt_translate_gcp_options(varargin)

% Arguments can either be a list of key, value pairs, or a struct.
% In latter case, convert to a list of pairs.  We always return a cell array.
if length(varargin) == 1 && isa(varargin{1}, 'struct')
  args = struct2params(varargin{1});
else
  args = varargin;
end

[type,args] = gt_get_option_and_remove(args,'type','normal');
if strcmp(type,'normal')
  type = 'gaussian';
elseif strcmp(type,'count')
  type = 'poisson';
elseif strcmp(type,'binary')
  type = 'bernoulli';
end
c = {'type' type};
args = [args c];

[fsamp,args] = gt_get_option_and_remove(args,'fsamp',[]);
if ~isempty(fsamp)
  if length(fsamp) == 1
    f = fsamp(1);
    c = {'fnzs', f, 'fzs', f};
  else
    f1 = fsamp(1);
    f2 = fsamp(2);
    c = {'fnzs', f1, 'fzs', f2};
  end
  args = [args c];
end
[gsamp,args] = gt_get_option_and_remove(args,'gsamp',[]);
if ~isempty(gsamp)
  if length(gsamp) == 1
    g = gsamp(1);
    c = {'gnzs', g, 'gzs', g};
  else
    g1 = gsamp(1);
    g2 = gsamp(2);
    c = {'gnzs', g1, 'gzs', g2};
  end
  args = [args c];
end
args = gt_rename_option(args,'sampler','sampling');
args = gt_rename_option(args,'maxfails','fails');
args = gt_rename_option(args,'epciters','epochiters');
args = gt_rename_option(args,'festtol','gcp-tol');
args = gt_rename_option(args,'beta1','adam-beta1');
args = gt_rename_option(args,'beta2','adam-beta2');
args = gt_rename_option(args,'epsilon','adam-eps');
args = gt_rename_option(args,'streaming_solver','streaming-solver');

end

function args = struct2params(S)
  % STRUCT2PARAMS Convert a structure to an interleaved array of key-value pairs for parameter parsing

  arg_names = fieldnames(S);
  arg_vals = struct2cell(S);
  args = reshape({arg_names{:}; arg_vals{:}}, [1,2*length(arg_names)]);
end
