function n = numArgumentsFromSubscript(X,s,indexingContext)
   switch s(1).type
    case '{}'
      error('Subscript cell reference cannot be used with sptensor.')

    case '.'
      switch s(1).subs
        case {'subs','indices'}
          n = 1;
        case {'vals','values'}
          n = 1;
        case 'size'
          n = 1;
        case {'perm','perms'}
          n = 1;
        case {'alg_params','params'}
          if length(s) == 1
            n = 1;
          else
            n = builtin('numArgumentsFromSubscript', X.alg_params, s(2:end), indexingContext);
          end
        otherwise
          error(['No such field: ', s(1).subs]);
      end

    case '()'
      n = 1;

    otherwise
      error('Incorrect indexing into sptensor.')
   end
end
