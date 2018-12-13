function A = import_sptensor_gt(fname)
%IMPORT_SPTENSOR_GT Import sparse tensor from file
%
%   A = IMPORT_SPTENSOR_GT(FNAME) imports a sparse tensor A from the file
%   named FNAME.
%
%   See also SPTENSOR_GT
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

%% Open file
fid = fopen(fname,'r');
if (fid == -1)
    error('Cannot open file %s',fname);
end

% Get and parse first line to discover dimensions
line = fgets(fid);
data = sscanf(line, '%f');
n = size(data,1)-1;

% Close and reopen file to reset fid
fclose(fid);
fid = fopen(fname,'r');

% Now read whole file
data = textscan(fid,[repmat('%u64',1,n) '%f']);
subs = cell2mat(data(1:n));
vals = data{n+1};

% Compute dimensions
sz = max(subs,[],1);

% Create tensor
A = sptensor_gt((subs-1)', vals, sz, 0);

% Close file
fclose(fid);