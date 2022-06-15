function varargout = subsref(t,s)
%SUBSREF Subscripted reference for a sparse tensor.
%
%   We can extract elements or subtensors from a sparse tensor in the
%   following ways.
%
%   Case 1a: y = X(i1,i2,...,iN), where each in is an index, returns a
%   scalar.
%
%   Case 1b: Y = X(R1,R2,...,RN), where one or more Rn is a range and
%   the rest are indices, returns a sparse tensor. The elements are
%   renumbered here as appropriate.
%
%   Case 2a: V = X(S) or V = X(S,'extract'), where S is a p x n array
%   of subscripts, returns a vector of p values.
%
%   Case 2b: V = X(I) or V = X(I,'extract'), where I is a set of p
%   linear indices, returns a vector of p values.
%
%   Any ambiguity results in executing the first valid case. This
%   is particularily an issue if ndims(X)==1.
%
%   S = X.subs returns the subscripts of the nonzero entries in X.
%
%   V = X.vals returns the values of the nonzero entries in X.
%
%   Examples
%   X = sptensor_gt([4,4,4;2,2,1;2,3,2],[3;5;1],[4 4 4]);
%   X(1,2,1) %<-- returns zero
%   X(4,4,4) %<-- returns 3
%   X(3:4,:,:) %<-- returns 2 x 4 x 4 sptensor
%   X(2,:,:) %<-- returns a 2 x 2 tensor
%   X([1,1,1;2,2,1]) %<-- returns a vector of 2 elements
%   X = sptensor_gt([6;16;26],[1;1;1],30);
%   X([1:6]') %<-- extracts a subtensor
%   X([1:6]','extract') %<-- extracts a vector of 6 elements
%
%   See also SPTENSOR_GT.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

switch s(1).type

    case '{}'
        error('Subscript cell reference cannot be used with sptensor.')

    case '.'
        switch s(1).subs
            case {'subs','indices'}
                varargout{1} = tt_subsubsref(double(t.subs'+1), s);
            case {'vals','values'}
                varargout{1} = tt_subsubsref(t.vals, s);
            case 'size'
                varargout{1} = tt_subsubsref(double(t.size), s);
            case {'perm','perms'}
                varargout{1} = tt_subsubsref(double(t.perm'+1), s);
            case {'alg_params','params'}
                %a = tt_subsubsref(t.alg_params, s);
                if length(s) == 1
                  varargout{1} = t.alg_params;
                else
                  % Provide support for t.alg_params{:}.
                  % It isn't clear to me if this correctly supports other uses
                  n = builtin('numArgumentsFromSubscript', t.alg_params, s(2:end), matlab.mixin.util.IndexingContext.Assignment);
                  varargout = cell(n,1);
                  [varargout{:}] = builtin('subsref', t.alg_params, s(2:end));
                end
            otherwise
                error(['No such field: ', s(1).subs]);
        end
        return;

    case '()'

        % *** CASE 1: Rectangular Subtensor ***
        if (numel(s(1).subs) == ndims(t)) && ...
                ~isequal(s(1).subs{end},'extract')

            % Extract the subdimensions to be extracted from t
            region = s(1).subs;

            % Pare down the list of subscripts (and values) to only
            % those within the subdimensions specified by region.
            loc = subdims(region, t);
            subs = t.subs(:,loc)'+1;
            vals = t.vals(loc);

            % Find the size of the subtensor and renumber the
            % subscripts
            [subs, sz] = renumber(subs, size(t), region);

            % Determine the subscripts
            newsiz = [];                % (future) new size
            kpdims = [];                % dimensions to keep
            rmdims = [];                % dimensions to remove

            % Determine the new size and what dimensions to keep
            for i = 1:length(region)
                if ischar(region{i}) && (region{i} == ':')
                    newsiz = [newsiz size(t,i)];
                    kpdims = [kpdims i];
                elseif numel(region{i}) > 1
                    newsiz = [newsiz numel(region{i})];
                    kpdims = [kpdims i];
                else
                    rmdims = [rmdims i];
                end
            end

            % Return a single double value for a zero-order sub-tensor
            if isempty(newsiz)
                if isempty(vals)
                    a = 0;
                else
                    a = vals;
                end
                varargout{1} = a;
                return;
            end

            % Assemble the resulting sparse tensor
            if isempty(subs)
                a = sptensor_gt([],[],sz(kpdims));
                a.alg_params = t.alg_params;
            else
                a = sptensor_gt(subs(:,kpdims), vals, sz(kpdims));
                a.alg_params = t.alg_params;
            end
            varargout{1} = a;
            return;
        end

        % Case 2: EXTRACT

        % *** CASE 2a: Subscript indexing ***
        if size(s(1).subs{1},2) == ndims(t)

            % extract array of subscripts
            srchsubs = s(1).subs{1};

       % *** CASE 2b: Linear indexing ***
        else

            % Error checking
            if numel(s(1).subs) ~= 1
                error('Invalid indexing');
            end

            idx = s(1).subs{1};
            if ndims(idx) ~=2 || size(idx,2) ~= 1
                error('Expecting a column index');
            end

            % extract linear indices and convert to subscripts
            srchsubs = tt_ind2sub(size(t),idx);

        end

        a = extract(t,srchsubs);
        a = tt_subsubsref(a,s);
        varargout{1} = a;

        return;

    otherwise
        error('Incorrect indexing into sptensor.')
end
