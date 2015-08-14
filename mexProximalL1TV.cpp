//[X,X_norms,gap,Qtv] = mexProximalL1TV(Y,lam1,lam2,idx,pos,wgt,Qtv_init)
//
//Mex function to solve the proximal operator of a matrix regularized by 
//the L1 and total variation norms with the optional constraint 
//that X>=0.  Specifically, it solves:
//
// argmin_{X} 0.5*|Y-X|_F^2 + sum_i (lam1(i)*|Xi|_1 + lam2(i)*|G*Xi|_1)
//
// (optionally) subject to  
//      X>=0
//
//where Xi denotes the i'th column of X
//
//G is a matrix that takes the weighted difference between pairs of entries
//in the columns of X.  Because G is often very large and sparse the
//function takes as input a N x 2 array that lists the pairs of indexes to
//take the difference between and an optional N x 1 vector that defines the
//weight to assign each difference between pairs of elements. For example,
//
// z = G*x;
//
// for i=1:length(z)
//    z2(i) = wgt(i)*(x(idx(i,1))-x(idx(i,2)))
// end
//
// would give identical vectors (z==z2)
//
//The algorithm proceeds by solving the dual formulation of the problem, 
//which is given by
//
//  min |Y-G'*Qtv-Q1|_F^2
//
//   subject to |Q1|<=lam1 and |Qtv|<=lam2
//
//   (and if X>=0 Y-G'*Qtv-Q1 >= 0)
//
//This actually simplifies by solving the dual
//  min |Y-G'*Qtv|_F^2
//
// then X = SoftThreshold(Y-G'*Qtv)
// or   X = PosSoftThreshold(Y-G'*Qtv) if X>=0 is active
//
//Inputs:
// Y - Data matrix
// lam1 - Vector of L1 regularization parameters.  Must be non-negative and
//          have length = number of columns in Y
// lam2 - Total variation regularizations.  Same conditions as lam1.
// idx - N x 2 int32 array which defines the pairs of elements in the 
//       columns of X to take the difference between (see above).  The 
//       function assumes the indices are 1-indexed.
//       (Matlab style indexing).  If no total variation term is wanted, 
//       then int32(zeros(0,2)) is a valid input.
// pos - Logical variable.  If true, then requires the solution to be 
//       nonnegative, otherwise no sign constraint is imposed.  If this is
//       not provided then the default is false.
// wgt - Optional vector describing the weight to assign the difference 
//       between pairs of elements in the columns of X (see above).  If
//       is not provided then a weight of 1 is used by default.
// Qtv_init - Optional initialization of the total variation dual
//            variables.  In iterative methods, such as proximal algorithms
//            small updates to the variables means that the dual solution 
//            should provide a much better restarting point than a default
//            starting initialization. If this is not provided the dual
//            variables are initialized to all zeros.
//
//
//Outputs:
// X - Solution matrix
// X_norms - 2 x size(Y,2) matrix containing the norms of the columns of X  
//          X_norms(1,i)= |X(:,i)|_1
//          X_norms(2,i)= |G*X(:,i)|_1
// gap - Final duality gap associated with each column of X.
// Qtv - The dual variables associated with total variation norm.  If space
//       exists in memory to save these during an iterative algorithm it
//       reduces the number of iterations to converge to a solution. 
//
//
// Ben Haeffele - Oct 2013

#ifndef MEX
#define MEX
#include "mex.h"
#endif

#include "L1TVprox.h"

//Main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, 
        const mxArray *prhs[]) {
    
    //maximum number of iterations
    int maxIter = 1000;
    
    //Stopping thresholds for the duality gap.
    double gapThreshRel = 1.0e-4;
    double gapThreshAbs = 1.0e-15;
    
    void * vars_created[MAX_NUM_VAR];
    int num_vars = 0;
    
    int mY = 0; //size of Y
    int nY = 0;
    int nIdx = 0; //number of rows in idx array
    
    int i = 0;
    int r = 0;
    int iterCount = 0;
    int col_idx = 0;
    
    double * lam1;
    double * lam2;
    
    bool makepos;  //positivity flag 
    mxLogical * temp_pos;
    
    double gap_init = 0; //variable to store initial duality gap
    double cur_gap = 0; //current value of duality gap
    
    char * dir_flag;
    proxBOOL sum_dir;
    
    mxArray * X_out;
    mxArray * gap_out;
    mxArray * X_norms_out;
    mxArray * Qtv_out;
        
    double * X;
    double * Xtv;
    double * Y;
    double * X_norms;
    double * Qtv;
    double * Qsum;
    double * wgt;
    double * gap;
    double * Qtv_init;
    int * idx;
    
    //Check arguments
    if(nrhs<4) {
        mexErrMsgTxt("At least 4 input arguments are required");
    }
    else;
    
    if(nlhs<1) {
        mexErrMsgTxt("No output argument specified");
    }
    else;
    
    for(i=0; i<nrhs; i++) {
        if(mxGetNumberOfDimensions(prhs[i])>2) {
            mexErrMsgTxt("Inputs with >2 dimensions are not allowed");
        }
        else;
    }
    
    if(!mxIsDouble(prhs[0])) {
        mexErrMsgTxt("Y must have type 'double'");
    }
    else;
    
    if(!mxIsDouble(prhs[1])) {
        mexErrMsgTxt("lam1 must have type 'double'");
    }
    else;
    
    if(!mxIsDouble(prhs[2])) {
        mexErrMsgTxt("lam2 must have type 'double'");
    }
    else;
    
    if(mxGetN(prhs[3])!=2) {
        mexErrMsgTxt("idx must have dimension N x 2");
    }
    else;
    
    if(!mxIsInt32(prhs[3])) {
        mexErrMsgTxt("idx must have type 'int32'");
    }
    else;
    
    //**********************
    //Read input parameters
       
    mY = mxGetM(prhs[0]);
    nY = mxGetN(prhs[0]);
    
    if((mxGetNumberOfElements(prhs[1])!=nY) || 
            (mxGetNumberOfElements(prhs[2])!=nY)) {
        mexErrMsgTxt("lam1 and lam2 have length = size(Y,2)");
    }
    else;
    
    Y = (double *) mxGetData(prhs[0]);
    lam1 = (double *) mxGetData(prhs[1]);
    lam2 = (double *) mxGetData(prhs[2]);
    
    for(i=0; i<nY; i++) {
        if((lam1[i]<0)||(lam2[i]<0)) {
            mexErrMsgTxt("lam1 and lam2 must be non-negative");
        }
        else;
    }
    
    nIdx = mxGetM(prhs[3]);    
    idx = (int *) mxGetData(prhs[3]);
    
    //check that the values of idx point to valid locations
    for(i=0; i<nIdx; i++) {
        if((idx[i]<1)||(idx[i]>mY)||(idx[i+nIdx]<1)||(idx[i+nIdx]>mY)) {
            mexErrMsgTxt("The values of 'idx' must be in the range [1,size(Y,1)]");
        }
        else;
    }
    
    //check if a positivity constraint is provided
    if(nrhs>4) {
        if(mxGetNumberOfElements(prhs[4])!=1) {
            mexErrMsgTxt("pos must be a single element");
        }
        else;
        
        if(!mxIsLogical(prhs[4])) {
            mexErrMsgTxt("pos must be logical");
        }
        else;
        
        temp_pos = mxGetLogicals(prhs[4]);
        makepos = temp_pos[0];
    }
    else {
        makepos = false;
    }
        
    //check if a wgt variable was provided
    if(nrhs>5) {
        if(mxGetNumberOfElements(prhs[5])!=nIdx) {
            mexErrMsgTxt("length(wgt) must equal size(idx,1)");
        }
        else;
        
        if(!mxIsDouble(prhs[5])) {
            mexErrMsgTxt("'wgt' must have type 'double'");
        }
        else;
        
        wgt = (double *) mxGetData(prhs[5]);
    }
    else {
        //Allocate memory for weight vector and initialize it with 1's
        wgt = (double *) CreateVar(nIdx,sizeof(double),
                vars_created,num_vars);
        
        for(i=0; i<nIdx; i++)
            wgt[i] = 1;
    }
    
    //check if initialization for Qtv was provided
    if(nrhs>6) {
        if(mxGetNumberOfElements(prhs[6])!=nIdx*nY) {
            mexErrMsgTxt("Qtv_init much have size(idx,1)*size(Y,2) elements");
        }
        else;
    
        if(!mxIsDouble(prhs[6])) {
            mexErrMsgTxt("'Qtv_init' must have type 'double'");
        }
        else;
        
        Qtv_init = (double *) mxGetData(prhs[6]);
    }
    else;   
        
    
    //Allocate memory for variables
    
    Xtv = (double *) CreateVar(mY,sizeof(double),vars_created,num_vars);
        
    //Assign output variables
    
    X_out = mxCreateDoubleMatrix(mY,nY,mxREAL);
    plhs[0] = X_out;
    X = (double *) mxGetData(X_out);
    
    if(nlhs>1) {
        X_norms_out = mxCreateDoubleMatrix(2,nY,mxREAL);
        plhs[1] = X_norms_out;
        X_norms = (double *) mxGetData(X_norms_out);
    }
    else {
        X_norms = (double *) CreateVar(2*nY,sizeof(double),
                vars_created,num_vars);
    }
    
    if(nlhs>2) {
        gap_out = mxCreateDoubleMatrix(1,nY,mxREAL);
        plhs[2] = gap_out;
        gap = (double *) mxGetData(gap_out);
    }
    else;
    
    if(nlhs>3) {
        Qtv_out = mxCreateDoubleMatrix(nIdx,nY,mxREAL);
        plhs[3] = Qtv_out;
        Qtv = (double *) mxGetData(Qtv_out);
    }
    else {
        Qtv = (double *) CreateVar(nIdx,sizeof(double),
                vars_created,num_vars);
    }
    
//*************************************************************************
//Start looping through the columns of Y
//*************************************************************************
    
    for(col_idx=0; col_idx<nY; col_idx++) {
    
        //Initialize X to be equal to Y   
        for(i=0; i<mY; i++)
            X[col_idx*mY+i] = Y[col_idx*mY+i];
    
        //If we have an initialization for Qtv, then set it here
        if(nrhs>6) {
            if(nlhs>3) {
                InitTV(&X[col_idx*mY],&Qtv[col_idx*nIdx],
                    &Qtv_init[col_idx*nIdx],idx,wgt,lam2[col_idx],
                    mY,1,nIdx);
            }
            else {
                InitTV(&X[col_idx*mY],Qtv,
                    &Qtv_init[col_idx*nIdx],idx,wgt,lam2[col_idx],
                    mY,1,nIdx);
            }
        }
        else {
            
            //If we don't have an initialization then set it to be 0.
            if(nlhs>3) {
                for(i=0; i<nIdx; i++)
                    Qtv[col_idx*nIdx+i] = 0;
            }
            else {
                for(i=0; i<nIdx; i++)
                    Qtv[i] = 0;
            }
        }
        
        //Initialize Xtv to X
        for(i=0; i<mY; i++)
            Xtv[i] = X[col_idx*mY+i];
          
        //Calculate initial column norms of Y
        
        X_norms[2*col_idx] = CalculateL1Norm(&Y[col_idx*mY],mY);
        X_norms[2*col_idx+1] = CalculateTVNorm(&Y[col_idx*mY],idx,wgt,
                mY,1,nIdx);
    
        //Initial duality gap is just the sum of the column norms of Y
        //(this isn't true if an initialization for Qtv is provided, but this
        //is only used for duality gap stopping conditions)
    
        gap_init = lam1[col_idx]*X_norms[2*col_idx] + 
                lam2[col_idx]*X_norms[2*col_idx+1];
    
        if(gap_init<=0) {
            if(makepos) {
                PosSoftThresholdMatrix(Xtv,&X[col_idx*mY],0,mY);
                X_norms[2*col_idx] = CalculateL1Norm(&X[col_idx*mY],mY);
                X_norms[2*col_idx+1] = CalculateTVNorm(&X[col_idx*mY],
                        idx,wgt,mY,1,nIdx);
            }
            else;
            continue;
        }
        else;
        
/*************************************************************************/
//Here we start the actual computation
/*************************************************************************/
    
        for(iterCount=0; iterCount<maxIter; iterCount++) {
        
            //Update dual TV variables
            if(nlhs>3) {
                UpdateTVDual(Xtv,&Qtv[col_idx*nIdx],idx,wgt,
                    lam2[col_idx],mY,1,nIdx);
            }
            else {
                UpdateTVDual(Xtv,Qtv,idx,wgt,
                    lam2[col_idx],mY,1,nIdx);
            }
            
            //Soft threshold for the l1 norm
            if(makepos) {
                PosSoftThresholdMatrix(Xtv,&X[col_idx*mY],
                        lam1[col_idx],mY);
            }            
            else {
                SoftThresholdMatrix(Xtv,&X[col_idx*mY],
                        lam1[col_idx],mY);
            }
                
            X_norms[2*col_idx] = CalculateL1Norm(&X[col_idx*mY],mY);
            X_norms[2*col_idx+1] = CalculateTVNorm(&X[col_idx*mY],idx,wgt,
                mY,1,nIdx);
           
            cur_gap = lam1[col_idx]*X_norms[2*col_idx] + 
                    lam2[col_idx]*X_norms[2*col_idx+1];
        
            cur_gap += CalculateFro2Norm(&X[col_idx*mY],mY) - 
                    DotProd(&X[col_idx*mY],&Y[col_idx*mY],mY);
        
            if(nlhs>2)
                gap[col_idx] = cur_gap;
            else;
        
            //Calculate stopping criteria
            if(((cur_gap/gap_init) < gapThreshRel) ||
                    (cur_gap < gapThreshAbs)) {
            //If we get here then we've converged to within our tolerances
                break;
            }
            else;
            
        }
        
        if(iterCount>=maxIter)
            mexWarnMsgIdAndTxt("mxProxL1TV:max_iter","Function reached the maximum number of iterations");
        else;
    }
    
    CleanupMem(vars_created,num_vars);
    return;
}