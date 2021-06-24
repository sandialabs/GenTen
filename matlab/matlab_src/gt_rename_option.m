function args = gt_rename_option(args,name,new_name)
% Rename an option

n = length(args);
for i=1:n
  if isa(args{i}, 'char') && strcmp(args{i},name)
    args{i} = new_name;
    break;
  end
end
