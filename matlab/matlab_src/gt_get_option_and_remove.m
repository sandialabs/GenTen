function [v,args] = gt_get_option_and_remove(args,name,dflt)
% Find an option with the given name and remove it from the list

n = length(args);
v = dflt;
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i},name)
    v = args{i+1};
    args(i:i+1) = [];
    break;
  end
end
