function A = import_sptensor_gt(fname)
%IMPORT_SPTENSOR_GT Import sparse tensor from file
%
%   A = IMPORT_SPTENSOR_GT(FNAME) imports a sparse tensor A from the file
%   named FNAME.
%
%   See also SPTENSOR_GT

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
data = fscanf(fid,'%f',[n+1,Inf]);

% Compute dimensions
sz = max(data(1:n,:),[],2);

% Create tensor
A = sptensor_gt(data(1:n,:)',data(n,:)',sz');

% Close file
fclose(fid);