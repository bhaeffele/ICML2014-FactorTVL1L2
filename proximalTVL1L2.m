function [X,normX,Qtv] = proximalTVL1L2(Y,lam,k,idx,pos,Qtv_init)
%[X,normX,Qtv] = proximalTVL1L2(Y,lam,k,idx,pos,Qtv_init)
%
%Solves the problem
%
% argmin_X 1/2*|X-Y|_F^2 + sum_i(lam(i)*|Xi|_k(i))
%
%where |Xi|_k(i) = k(i,1)*|Xi|_1 + k(i,2)*|G*Xi|_1 + k(i,3)*|Xi|_2
%
%and Xi denotes the i'th column of X and G is a matrix that takes the
%difference between elements of Xi.
%
% Note: Passing a value of lam<=0 will return X=Y and calculate the norm of
% the columns of Y.
%
%Parameters:
%   Y - Data
%   
%   lam - Regularization parameter.  Can either be a scalar  
%       or an N element vector, where N is the number of columns in Y.  
%       If lam is a vector then lam(i) is used for the i'th column of Y,
%       while if lam is a scalar then the same value is used for all
%       columns.
%
%   k - Regularization parameters.  Can either be a 1x3 vector or a Nx3
%       array.  If k is an array, then the values of k are given as above.
%       If k is a vector, then the same values are used for all columns.
%
%   idx - Q x 2 array listing the pairs of indexes of X to take the
%       difference between.  Should be int32
%
%   pos - If true, then values are constrained to be nonnegative.
%
%   Qtv_init - Initialization for total variation dual variables.(Optional)
%
%Outputs:
%   X - Solution
%
%   normX - 1 x N vector containing the value of |Xi|_k(i) for each column.
%
%   Qtv - Total variation dual variables
%
% Ben Haeffele: Aug - 2013
%
%   Aug 2015 - Fixed bug where the non-negativity constraints wouldn't be
%              enforced if k(1) and k(2) were 0.

[P,num_col] = size(Y);

X = zeros(P,num_col);
normX = zeros(1,num_col);

if length(lam)==1
    lam = lam*ones(num_col,1);
else
    lam = lam(:);
end

if size(k,1)==1
    k = repmat(k,num_col,1);
end

if nargout>=3
    Qtv = zeros(size(idx,1),num_col);
end

if exist('Qtv_init','var')
    wgt = ones(size(idx,1),1);
end

if sum(sum(k(:,1:2)))<=0
    %Only using the L2 norm.
    if pos
        X = Y.*(Y>0);
    else
        X = Y;
    end
    normX = zeros(1,size(X,2));
elseif sum(k(:,2))<=0
    %No TV norm but there is an L1 norm.
    if pos
        X = max(Y-repmat((k(:,1).*lam)',size(Y,1),1),0);
    else
        X = sign(Y).*max(abs(Y)-repmat((k(:,1).*lam)',size(Y,1),1),0);
    end
    
    normX = sum(abs(X)).*k(:,1)';
else
    if nargout>=3
        if exist('Qtv_init','var')
            [X,temp_norm,~,Qtv] = mexProximalL1TV(Y,lam.*k(:,1),lam.*k(:,2),...
                idx,pos,wgt,Qtv_init);
        else
            [X,temp_norm,~,Qtv] = mexProximalL1TV(Y,lam.*k(:,1),lam.*k(:,2),...
                idx,pos);
        end
    else
        if exist('Qtv_init','var')
            [X,temp_norm] = mexProximalL1TV(Y,lam.*k(:,1),lam.*k(:,2),...
                idx,pos,wgt,Qtv_init);
        else
            [X,temp_norm] = mexProximalL1TV(Y,lam.*k(:,1),lam.*k(:,2),...
                idx,pos);
        end            
    end

    normX = diag(k(:,1:2)*temp_norm)';
end

%Now project onto the L2 ball

L2 = sqrt(sum(X.^2)); %calculate L2 norms
idx_nz = (L2>0) & (k(:,3)>0)'; %find indexes we need to update

if sum(idx_nz)
    normX(idx_nz) = normX(idx_nz) + k(idx_nz,3)'.*L2(idx_nz); %add L2 norm
    scl = 1-min(L2(idx_nz),lam(idx_nz)'.*k(idx_nz,3)')./L2(idx_nz);

    %scale columns of X to finish projection onto L2 ball
    X(:,idx_nz) = bsxfun(@times,X(:,idx_nz),scl);
    normX(idx_nz) = normX(idx_nz).*scl; %update norms
end