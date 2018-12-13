function export(A, fname)
%EXPORT Export Sptensor_gt to a file.
%
%   EXPORT(A,FNAME) exports an sptensor_gt object A to the
%   file named FNAME in plain ASCII text.  The format for a 4 x 3 x 2
%   sptensor_gt with 10 nonzeros is as follows...
%      i1 j1 k1 <value of A(i1,j1,k1)>
%      i2 j2 k2 <value of A(i2,j2,k2)>
%      ...
%      i10 j10 k10 <value of A(i10,j10,k10)>
%
%   See also SPTENSOR_GT, IMPORT_SPTENSOR_GT
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.

%% Open file
fid = fopen(fname,'w');
if (fid == -1)
    error('Cannot open file %s',fname);
end

%% Export the object

for i = 1:nnz(A)
  for s = 1:length(A.size)
    fprintf(fid,'%d ', A.subs(s,i)+1);
  end
  fprintf(fid,'%.16e\n',A.vals(i));
end

%% Close file
fclose(fid);
