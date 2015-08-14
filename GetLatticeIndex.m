function idx = GetLatticeIndex(m,n,sz)
%idx = GetLatticeIndex(m,n)
%idx = GetLatticeIndex(m,n,sz)
%
%Returns a cell array containing the indecies of neighboring pixels.  Each
%cell contains the neighbor pairs for one direction.
% idx{1} - horizontal pixel pairs
% idx{2} - vertical pixel pairs
% idx{3} - diagonal pixel pairs (down and to the right)
% idx{4} - diagonal pixel pairs (down and to the left)
%
% if sz>1 then the pattern is repeated, but for larget pixel steps i.e.
% idx{5} - horizontal pixel pairs 2 pixels apart
% idx{6} - vertical pixel pairs 2 pixels apart
% etc...
%
%m - image height
%n - image width
%az - size of neighborhood (radius). Default is 1 if not provided.

if ~exist('sz','var')
    sz = 1;
end

idx = cell(4*sz,1);

idx_array = reshape(1:(m*n),m,n);

%horizontal groups
for i=1:sz
    k = 2^(i-1);
    
    idx_p = [];
    idx_n = [];
    temp = idx_array(:,1:n-k);
    idx_p = [idx_p(:); temp(:)];
    
    temp = idx_array(:,1+k:n);
    idx_n = [idx_n(:); temp(:)];
    
    idx{(i-1)*4+1} = [idx_p(:), idx_n(:);];
end

%vertical groups
for i=1:sz
    k = 2^(i-1);
    
    idx_p = [];
    idx_n = [];
    temp = idx_array(1:m-k,:);
    idx_p = [idx_p(:); temp(:)];
    
    temp = idx_array(k+1:m,:);
    idx_n = [idx_n(:); temp(:)];
    idx{(i-1)*4+2} = [idx_p(:), idx_n(:);];
end

%diagonal group 1 (going down and to the right)
for i=1:sz
    k = 2^(i-1);
    
    idx_p = [];
    idx_n = [];
    temp = idx_array(1:m-k,1:n-k);
    idx_p = [idx_p(:); temp(:)];
    
    temp = idx_array(k+1:m,k+1:n);
    idx_n = [idx_n(:); temp(:)];
    idx{(i-1)*4+3} = [idx_p(:), idx_n(:);];
end

%diagonal group 2 (going up and to the right)
for i=1:sz
    k = 2^(i-1);
    
    idx_p = [];
    idx_n = [];
    temp = idx_array(k+1:m,1:n-k);
    idx_p = [idx_p(:); temp(:)];
    
    temp = idx_array(1:m-k,k+1:n);
    idx_n = [idx_n(:); temp(:)];
    idx{(i-1)*4+4} = [idx_p(:), idx_n(:);];
end

end

