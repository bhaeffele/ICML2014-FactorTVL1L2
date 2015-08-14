function [A,Z,B,obj,A_norm,Z_norm,B_norm,L_stats] = FactorTVL1L2_v1(X,Da,Db,param,num_iter)
%[A,Z,B,obj,A_norm,Z_norm,B_norm,L_stats] = FactorTVL1L2_v1(X,Da,Db,param,num_iter)
%
%The optimization problem solved is given as:
%
% argmin_{A,Z,B} 1/2*|X-Da*A*Z'+Db*B'|_F^2 + lam*r(A,Z) + lam_B*r(B)
%
%where r(A,Z) is a norm on the product A*Z'.
%Here we use the regularization:
%
% r(A,Z) = sum_i (|Ai|_kA + |Zi|_kZ)
%
%where Ai and Zi denote the i'th columns of A and Z, respectively, and
%
%  |Qi|_k = k(1)*|Qi|_1 + k(2)*|G*Qi|_1 + k(3)*|Qi|_2
%
%for a matrix G which takes the difference between elements of Qi.
%Alternatively, |G*Qi|_1 is the total variation pseudo-norm of Qi.
%Qualitatively, the three norms have the effects:
%
%   |Qi|_1 - Encourage column to be sparse (small number of non-zeros).
%   |G*Qi|_1 - Encourage neighboring elements of Qi to have similar values
%              (Total variation on column using graph defined by matrix G).
%   |Qi|_2 - Shrinks the total size of the column, encouraging a small
%            number of columns to be used in the factorization (and thus a
%            small rank of the solution A*Z').
%
% B is an optional intercept term that is regularized seperately via:
%
% r(B) = sum_i |B_i|_kB
%
%Parameters:
%
%   X - Data matrix.
%
%   Da - Dictionary matrix.  If the identity is to be used, then [] is 
%        acceptable (which is useful if size(A,1) is very large).
%   
%   Db - Intercept dictionary.  Using [] will result in no intercept.
%
%   param.lam - Regularization parameter.
%
%   param.lam_B - Regularization parameter for intercept. (Optional -
%       Default 0)
%
%   param.posA 
%   param.posZ
%   param.posB - If true, then force A, Z or B, repectively to be
%                nonnegative (Optional - Default false).
%
%   param.kA - 1 x 3 vector.  Row defines the values of k to
%       use for the regularization on the columns of A. Must be
%       nonnegative.
%
%   param.kZ - 1 x 3 array.  Same as kA, but for the columns of Z.
%
%   param.kB - 1 x 3 array.  Same as kA, but for the columns of the
%       intercept B. (Optional - Default [0 0 0]).
%
%   param.idx - Cell array that defines groups of elements to take the
%       difference between for the total variation regularization.
%       The contents of each cell should be a M x 2
%       array which defines pixel pairs.  If this is not provided then idx
%       is initialized to not take a difference between any pixels.
%       (Optional).
%
%   param.idx_indexA - Vector. Should contain the indexes of the cells of 
%       param.idx to use to take for the columns of A. If this isn't given, 
%       then this is initialized to be all of the cells in idx if kA(2)>0,
%       or no cells if kA(2)=0. (Optional)
%
%   param.idx_indexZ - Same as idx_indexA but for the columns of Z.
%       (Optional)
%
%   param.idx_indexB - Same as idx_indexA but for the columns of B.
%       (Optional)
%
%   param.A_init - Initialization for the matrix A. If this isn't provided,
%       then A is initialized to be random columns sampled from an identity
%       matrix, where the number of columns is given by param.rank.
%       (Optional)
%
%   param.rank - If A_init and Z_init are not provided, then this sets the
%       rank of the solution to solve for.  If this is not provided, then
%       the variables are initialized to be full rank.
%       (Optional)
%
%   param.Z_init - Initialization for the matrix Z. (Optional).
%
%   param.B_init - Initialization for the intercept matrix B.  If this is
%       not provided, then a least squares fit is used. (Optional).
%
%   param.start_A - If true then A is the first variable optimized over.
%       If false, then Z is optimized first.  (Optional - Default false).
%
%   param.display - If true, then the iteration number is printed and the
%       matrix A is displayed at each iteration.  (Optional - Default true)
%
%   param.Da_norm - Matrix norm of Da.  This option is provided since
%       for large D calculating |Da|_2 (largest singular value) is
%       infeasible, so an estimate of the
%       norm must be provided.  If this parameter isn't provided then the
%       function will attempt to calculate it (which can take a very long
%       time for large Da). (Optional)
%
%   param.Db_norm - Matrix norm of Db.  Same as Da_norm. (Optional).
%
%   param.save_dual - If true, then the dual variables of the proximal
%   operators will be saved for an initialization of the next iteration.  
%   This can sometimes speed up the computation if there is space in memory  
%   (Optional - Default false)
%
%   num_iter - Number of optimization iterations to perform.
%
%Outputs:
%   A - Solution matrix.
%
%   Z - Solution matrix.
%
%   B - Solution matrix.
%
%   obj - Vector containing the value of the objective function at each
%       iteration.
%
%   A_norm - Vector containing the norms of the columns of A.
%
%   Z_norm - Vector containing the norms of the columns of Z.
%
%   B_norm - Vector containing the norms of the columns of B (if kB is not
%       given, then this is 0).
%
%   L_stats - Matrix containing the lipschitz constants. Constants for A
%       are in the first column, constants for Z/B in the second.
%
%The optimization is solved by block coordinate descent over A and Z
%(starting with A) using a proximal update combined with a linear
%extrapolation.  More details can be found here:
%
%Y. Xu and W. Yin, "A Block Coordinate Descent Method for Regularized
%  Multi-Convex Optimization with Applications to Nonnegative Tensor
%  Factorization and Completion." 2012.
%
% Ben Haeffele - Oct 2013

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%**************************************************************************
%Parse input parameters
%**************************************************************************

%Get data size
[mX,nX] = size(X);

%Check what dictionaries we're using and if there's a filter
use_Da = numel(Da)>0;
use_Db = numel(Db)>0;

%Check for positivity parameters
if ~isfield(param,'posA') 
    param.posA = false; 
else
    param.posA = logical(param.posA);
end

if ~isfield(param,'posZ')
    param.posZ = false;
else
    param.posZ = logical(param.posZ);
end

if ~isfield(param,'posB')
    param.posB = false;
else
    param.posB = logical(param.posB);
end

%Figure out what size the variable matrices should be

%A
if use_Da
    szA = size(Da,2);
else
    szA = mX;
end

%Z
szZ = nX;

%B
if use_Db
    num_colB = size(Db,2);
    szB = nX;
else
    num_colB = 0;
    szB = 0;
end

%See if we've got variable initializations

%A
if isfield(param,'A_init')
    A = param.A_init;
    num_col = size(A,2);
    
    if size(A,1)~=szA
        error('Size of A_init is not consistent');
    end
end

%Z
if isfield(param,'Z_init')
    Z = param.Z_init;
    if exist('num_col','var')
        if size(Z,2)~=num_col
            error('Sizes of A_init and Z_init are not consistent');
        end
    else
        num_col = size(Z,2);
    end
    
    if size(Z,1)~=szZ
        error('Size of Z_init is not consistent');
    end
end

%B
if isfield(param,'B_init')
    B = param.B_init;
    
    if use_Db
        if (size(B,1)~=szB) || (size(B,2)~=num_colB)
            error('Size of B_init is not consistent');
        end
    else
        if numel(B)>0
            warning('B_init provided, but no intercept dictionary is being used');
        end
    end
end

%Make generic initializations for variables we weren't given
%initializations for

if ~isfield(param,'start_A')
    param.start_A = false;
end

if ~exist('num_col','var')
    %No info for either A or Z, so we need to know how any columns to use
    if isfield(param,'rank')
        num_col = param.rank;
    else
        num_col = min(szA,szZ);
    end
end

%A
if ~exist('A','var')
    A = zeros(szA,num_col);
    if ~param.start_A
        %We're going to optimize Z first, so initialize A to be sampled 
        %columns from an identity matrix
        if num_col<=szA
            idx_temp = randperm(szA,num_col);
        else
            idx_temp = [1:szA randi(szA,1,num_col-szA)];
        end
        
        for i=1:num_col
            A(idx_temp(i),i) = 1;
        end
    end
end

%Z
if ~exist('Z','var')
    Z = zeros(szZ,num_col);
    if param.start_A
       %We're optimizing A first, so initialize Z to be sampled columns
       %from an identity matrix
       if num_col<=szZ
           idx_temp = randperm(szZ,num_col);
       else
           idx_temp = [1:szZ randi(szZ,1,num_col-szZ)];
       end
           
       for i=1:num_col
           Z(idx_temp(i),i) = 1;
       end
    end
end

%B
if ~exist('B','var')
    if use_Db
        %We initialize B to just be a least squares fit, ignoring any
        %regularization.
        if use_Da
            B = (Db \ (X-Da*A*Z'))';
        else
            B = (Db \ (X-A*Z'))';
        end
    else
        B = [];
    end
end

%Initialize other variables if they aren't given

if ~isfield(param,'lam_B')
    param.lam_B = 0;
end
   
if ~isfield(param,'kB')
    if param.lam_B > 0
        warning('lam_B is > 0, but no kB parameters were given');
    end
    param.kB = [0 0 0];
end

if ~isfield(param,'display')
    param.display = true;
end

if ~isfield(param,'save_dual')
    param.save_dual = false;
end

if ~isfield(param,'idx')
    if param.kA(2)>0 || param.kZ(2)>0 || param.kB(2)>0
        warning('param.idx was not provided for total variation regularization');
    end
    param.idx{1} = zeros(0,2);
end

L_stats = zeros(num_iter,2);

obj = zeros(num_iter+1,1);

if param.kA(2)>0
    if isfield(param,'idx_indexA')
        indexA = int32(cell2mat(param.idx(param.idx_indexA)));
    else
        indexA = int32(cell2mat(param.idx));
    end
else
    indexA = int32(zeros(0,2));
end

if param.kZ(2)>0
    if isfield(param,'idx_indexZ')
        indexZ = int32(cell2mat(param.idx(param.idx_indexZ)));
    else
        indexZ = int32(cell2mat(param.idx));
    end
else
    indexZ = int32(zeros(0,2));
end

if param.kB(2)>0
    if isfield(param,'idx_indexB')
        indexB = int32(cell2mat(param.idx(param.idx_indexB)));
    else
        indexB = int32(cell2mat(param.idx));
    end
else
    indexB = int32(zeros(0,2));
end

if param.save_dual
        
    dualA = zeros(size(indexA,1),num_col);
    dualZ = zeros(size(indexZ,1),num_col);
    dualB = zeros(size(indexB,1),num_colB);
end

%**************************************************************************
%Setup preliminaries
%**************************************************************************

LA_old = 1;
LZ_old = 1;
t_old = 1;

Z_extrap = Z;
A_extrap = A;
B_extrap = B;

Z_old = Z;
A_old = A;
B_old = B;

if use_Da
    if isfield(param,'Da_norm')
        Da_norm = param.Da_norm;
    else
        Da_norm = norm(Da,2);
    end
else
    Da_norm = 1;
end

if use_Db
    if ~isfield(param,'Db_norm')
        Db_norm = norm(Db,2);
    else
        Db_norm = param.Db_norm;
    end
end

%Calculate the initial norms of the columns of A, B, and Z.  We use the
%proximal operator function for convenience.

[A,A_norm] = proximalTVL1L2(A,0,param.kA,indexA,param.posA);
[Z,Z_norm] = proximalTVL1L2(Z,0,param.kZ,indexZ,param.posZ);

if use_Db
    [B,B_norm] = proximalTVL1L2(B,0,param.kZ,indexB,param.posB);
else
    B_norm = 0;
end

Anrm_old = A_norm;
Znrm_old = Z_norm;
Bnrm_old = B_norm;

%Precalculate matrices to speedup gradient and objective function
%calculations.  It gets a little messy keeping track of the 4 different
%possible conditions.

const.Xfro = 0.5*norm(X,'fro')^2;
varbl.ZTZ = Z'*Z;

if use_Da
    const.DaTDa = Da'*Da;
    const.DaTX = Da'*X;
    varbl.DaTXZ = const.DaTX*Z;
    varbl.ADDA = A'*const.DaTDa*A;
else
    varbl.XZ = X*Z;
    varbl.ATA = A'*A;    
end

if use_Db
    const.DbTDb = Db'*Db;
    const.XTDb = X'*Db;
    varbl.BTZ = B'*Z;
else
    varbl.XZ = X*Z;
end

if use_Da && use_Db
    const.DaTDb = Da'*Db;
    varbl.DbTDaA = const.DaTDb'*A;
end
    
if use_Db && not(use_Da)
    varbl.DbTA = Db'*A;
end

cur_obj=CalcObj(A,Z,B,use_Da,use_Db,const,varbl,A_norm,Z_norm,B_norm,param);

obj(1,1) = cur_obj;

%**************************************************************************
%Start of the algorithm
%**************************************************************************

iter_count = 1;
do_A = false;

if param.display
    disp('Start');
end

while iter_count <= num_iter
    
    %%%%%%%%
    %Update for A
    %%%%%%%%
    
    if param.start_A || do_A
        
        %Calculate gradient of A.  If using an intercept and a dictionary
        %Da, this is given as
        %A_grad = Da'*(Da*A_extrap*Z'+Db*B'-X)*Z;
        %
        %Below is just to calculate this a bit faster depending on
        %which dictionaries we're using and to expoilt the structure of our 
        %factorized matrices to precalculate things.
        
        if use_Da
            if use_Db
                A_grad = const.DaTDa*A_extrap*varbl.ZTZ + ...
                    const.DaTDb*varbl.BTZ - varbl.DaTXZ;
            else
                A_grad = const.DaTDa*A_extrap*varbl.ZTZ-varbl.DaTXZ;
            end
        else
            if use_Db
                A_grad = A_extrap*varbl.ZTZ + Db*varbl.BTZ - varbl.XZ;
            else
                A_grad = A_extrap*varbl.ZTZ - varbl.XZ;
            end
        end
        
        %Calculate lipschitz constant for the gradient of A
        
        LA = norm(varbl.ZTZ,2)*Da_norm^2;
        
        A_proj = A_extrap-A_grad/LA;
        
        idx_nz = Z_norm>0;
        
        %Caclulate proximal operator of A
        if param.save_dual
            [A(:,idx_nz),A_norm(idx_nz),dualA(:,idx_nz)] = ...
            proximalTVL1L2(A_proj(:,idx_nz),param.lam*Z_norm(idx_nz)/LA,...
            param.kA,indexA,param.posA,dualA(:,idx_nz));
        else
            [A(:,idx_nz),A_norm(idx_nz)] = ...
            proximalTVL1L2(A_proj(:,idx_nz), ...
            param.lam*Z_norm(idx_nz)/LA,param.kA,indexA,param.posA);
        end
        
        A(:,not(idx_nz)) = 0;
        A_norm(not(idx_nz)) = 0;
        
        %Update the precalculated matrices
        
        if use_Da
            varbl.ADDA = A'*const.DaTDa*A;
        else
            varbl.ATA = A'*A;    
        end

        if use_Da && use_Db
            varbl.DbTDaA = const.DaTDb'*A;
        end

        if use_Db && not(use_Da)
            varbl.DbTA = Db'*A;
        end
                
        if param.display
            imagesc(A);
            drawnow;
        end
        
        if ~param.start_A
            %If we get here, then we've updated all the variables for this
            %iteration, so now calculate the objective function value and
            %extrapolate the variables for the next iteration.
            
            cur_obj = CalcObj(A,Z,B,use_Da,use_Db,const,varbl,A_norm, ...
                Z_norm,B_norm,param);
            
            obj(iter_count+1,1) = cur_obj;
            
            if cur_obj>=obj(iter_count)
            %Objective didn't decrease, so run again without extrapolation.
                A_extrap = A_old;
                Z_extrap = Z_old;
                B_extrap = B_old;
                
                A_norm = Anrm_old;
                Z_norm = Znrm_old;
                B_norm = Bnrm_old;
                
                A = A_old;
                Z = Z_old;
                B = B_old;
                
                if use_Da
                    varbl.DaTXZ = const.DaTX*Z;
                    varbl.ADDA = A'*const.DaTDa*A;
                else
                    varbl.XZ = X*Z;
                    varbl.ATA = A'*A;    
                end

                if use_Db
                    varbl.BTZ = B'*Z;
                else
                    varbl.XZ = X*Z;
                end

                if use_Da && use_Db
                    varbl.DbTDaA = const.DaTDb'*A;
                end

                if use_Db && not(use_Da)
                    varbl.DbTA = Db'*A;
                end
            else
                t = (1+sqrt(1+4*t_old^2))/2;
                
                w = (t_old-1)/t;
        
                wA = min(w,sqrt(LA_old/LA));
                A_extrap = A+wA*(A-A_old);

                wZ = min(w,sqrt(LZ_old/LZ));
                Z_extrap = Z+wZ*(Z-Z_old);
                B_extrap = B+wZ*(B-B_old);

                Z_old = Z;
                A_old = A;
                B_old = B;
                t_old = t;
                
                Anrm_old = A_norm;
                Znrm_old = Z_norm;
                Bnrm_old = B_norm;

                LA_old = LA;
                LZ_old = LZ;
            end
            
            if param.display
                disp(iter_count);
            end
            
            iter_count = iter_count+1;
                        
        end
    end
    
    if iter_count <= num_iter
        do_A = true;
        
        %%%%%%%%
        %Update for Z
        %%%%%%%%
        
        %Calculate gradient of Z. If using an intercept and a dictionary
        %Da, this is given as
        %
        % Z_grad = (Z_extrap*A'*Da'+B*Db'-X')*Da*A
        
        if use_Da
            if use_Db
                Z_grad = Z_extrap*varbl.ADDA+B*varbl.DbTDaA-const.DaTX'*A;
            else
                Z_grad = Z_extrap*varbl.ADDA-const.DaTX'*A;
            end
        else
            if use_Db
                Z_grad = Z_extrap*varbl.ATA+B*varbl.DbTA-X'*A;
            else
                Z_grad = Z_extrap*varbl.ATA-X'*A;
            end
        end
                           
        %Calculate lipchistz constant of the gradient of Z       
        
        if use_Da
            LZ = norm(varbl.ADDA,2);
        else
            LZ = norm(varbl.ATA,2);
        end
                      
        Z_proj = Z_extrap-Z_grad/LZ;
        
        idx_nz = A_norm>0;
        
        %Caclulate proximal operator of Z
        if param.save_dual
            [Z(:,idx_nz),Z_norm(idx_nz),dualZ(:,idx_nz)] = ...
            proximalTVL1L2(Z_proj(:,idx_nz),param.lam*A_norm(idx_nz)/LZ,...
            param.kZ,indexZ,param.posZ,dualZ(:,idx_nz));
        else
            [Z(:,idx_nz),Z_norm(idx_nz)] = ...
            proximalTVL1L2(Z_proj(:,idx_nz), ...
            param.lam*A_norm(idx_nz)/LZ,param.kZ,indexZ,param.posZ);
        end
        
        Z(:,not(idx_nz)) = 0;
        Z_norm(not(idx_nz)) = 0;
        
        %Update precalculated matrices
        
        varbl.ZTZ = Z'*Z;

        if use_Da
            varbl.DaTXZ = const.DaTX*Z;
        else
            varbl.XZ = X*Z;
        end

        %%%%%%%%
        %Update for B
        %%%%%%%%
        
        if use_Db
            
            %Calculate gradient of B.  Given by
            %
            % B_grad = (B*Db'+Z*A*Da'-X')*Db
            %
            
            if use_Da
                B_grad = B_extrap*const.DbTDb+Z*varbl.DbTDaA'-const.XTDb;
            else
                B_grad = B_extrap*const.DbTDb+Z*varbl.DbTA'-const.XTDb;
            end
            
            LB = Db_norm^2;
            
            B_proj = B_extrap-B_grad/LB;
            
            if param.save_dual
            [B,B_norm,dualB] = proximalTVL1L2(B_proj,param.lam_B/LB, ...
                param.kB,indexB,param.posB,dualB);
            else
             [B,B_norm] = proximalTVL1L2(B_proj,param.lam_B/LB, ...
                param.kB,indexB,param.posB);
            end
            
            %Update precalculated matrices
            
            varbl.BTZ = B'*Z;
        end
        
        if param.start_A            
            %If we get here, then we've updated all the variables for this
            %iteration, so now calculate the objective function value and
            %extrapolate the variables for the next iteration.
            
             cur_obj = CalcObj(A,Z,B,use_Da,use_Db,const,varbl,A_norm, ...
                Z_norm,B_norm,param);
            
            obj(iter_count+1,1) = cur_obj;
            
            if cur_obj>=obj(iter_count)
            %Objective didn't decrease, so run again without extrapolation.
                A_extrap = A_old;
                Z_extrap = Z_old;
                B_extrap = B_old;
                
                A_norm = Anrm_old;
                Z_norm = Znrm_old;
                B_norm = Bnrm_old;
                
                A = A_old;
                Z = Z_old;
                B = B_old;
                
                if use_Da
                    varbl.DaTXZ = const.DaTX*Z;
                    varbl.ADDA = A'*const.DaTDa*A;
                else
                    varbl.XZ = X*Z;
                    varbl.ATA = A'*A;    
                end

                if use_Db
                    varbl.BTZ = B'*Z;
                else
                    varbl.XZ = X*Z;
                end

                if use_Da && use_Db
                    varbl.DbTDaA = const.DaTDb'*A;
                end

                if use_Db && not(use_Da)
                    varbl.DbTA = Db'*A;
                end
            else
                t = (1+sqrt(1+4*t_old^2))/2;
                
                w = (t_old-1)/t;
        
                wA = min(w,sqrt(LA_old/LA));
                A_extrap = A+wA*(A-A_old);

                wZ = min(w,sqrt(LZ_old/LZ));
                Z_extrap = Z+wZ*(Z-Z_old);
                B_extrap = B+wZ*(B-B_old);

                Z_old = Z;
                A_old = A;
                B_old = B;
                t_old = t;
                
                Anrm_old = A_norm;
                Znrm_old = Z_norm;
                Bnrm_old = B_norm;

                LA_old = LA;
                LZ_old = LZ;
            end
            
            if param.display
                disp(iter_count);
            end
            
            iter_count = iter_count+1;
            
        end
    end
end

if param.display
    beep;
end

end

function obj = CalcObj(A,Z,B,use_Da,use_Db,const,varbl,A_norm,Z_norm, ...
    B_norm,param)
%Calculates the current value of the objective function

%The following is just a very verbose way of calculating
%
% 0.5*norm(X-Da*A*Z'-Db*B','fro')^2
%
%Splitting it into an expanded form saves some calculation time due to the
%structure of our factorized matrices and allows
%computations to be shared between the objective calculation and the
%gradient calculations.

obj = const.Xfro;

if use_Da
    obj = obj - varbl.DaTXZ(:)'*A(:) + ...
        0.5*(varbl.ADDA(:)'*varbl.ZTZ(:));
else
    obj = obj - varbl.XZ(:)'*A(:)+0.5*(varbl.ATA(:)'*varbl.ZTZ(:));
end

if use_Db
    BTB = B'*B;
    obj = obj-const.XTDb(:)'*B(:)+0.5*const.DbTDb(:)'*BTB(:);
    
    if use_Da
        obj = obj + varbl.DbTDaA(:)'*varbl.BTZ(:);
    else
        obj = obj + varbl.DbTA(:)'*varbl.BTZ(:);
    end
end

%now obj = 0.5*norm(X-Da*A*Z'-Db*B','fro')^2
%
%Add in the regularization terms

obj = obj+param.lam*sum(A_norm.*Z_norm)+param.lam_B*sum(B_norm);

end