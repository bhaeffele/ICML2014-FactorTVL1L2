//This file contains most the functions necessary to update various 
//variable types in the solution of proximal operators related to L1 and
//total variation minimization with various types of constraints.
//
//Ben Haeffele - Oct 2013

#include "L1TVprox.h"
#include "stdlib.h"

//See L1TVprox.h for most of the function interface definitions

void SoftThresholdMatrix(const double X[], double Y[], const double lam, 
        const int length) {
    
    for(int i=0; i<length; i++) 
        Y[i] = softthresh(X[i],lam);
}

void SoftThresholdMatrix(const double X[], double Y[], const double lam[],
        const int mX, const int nX) {
    
    for(int i=0; i<nX; i++)
        SoftThresholdMatrix(&X[i*mX],&Y[i*mX],lam[i],mX);
}

void PosSoftThresholdMatrix(const double X[], double Y[], const double lam, 
        const int length) {
    
    for(int i=0; i<length; i++)
        Y[i] = posthresh(X[i],lam);
}

void PosSoftThresholdMatrix(const double X[], double Y[], 
        const double lam[], const int mX, const int nX) {
    
    for(int i=0; i<nX; i++)
        PosSoftThresholdMatrix(&X[i*mX],&Y[i*mX],lam[i],mX);
}

void UpdateL1Dual(double X[], double Q[], const double lam, 
        const int length) {

    double XplusQ = 0;
    
    if(lam<=0)
        return;
    else;
       
    for(int i=0; i<length; i++) {
        XplusQ = X[i]+Q[i];
        Q[i] = sgn(XplusQ)*min(fabs(XplusQ),lam);
        X[i] = XplusQ-Q[i];
    }
}

void UpdateL1Dual(double X[], double Q[], const double lam[],
        const int mX, const int nX) {
    
    for(int i=0; i<nX; i++)
        UpdateL1Dual(&X[i*mX],&Q[i*mX],lam[i],mX);
}

void UpdateTVDual(double X[], double Q[], const int idx[],
        const double wgt[], const double lam, const int mX, const int nX,
        const int nIdx) {
    
    //X is mX x nX
    //Q is nIdx x nX
    //
    //idx should be 1 indexed (matlab style)
    
    double Qnew = 0;
    int idx_pos = 0;
    int idx_neg = 0;
    int idx_Q = 0;
    
    if(lam<=0)
        return;
    else;
    
    for(int c=0; c<nX; c++) {
        for(int i=0; i<nIdx; i++) {
            idx_Q = c*nIdx + i;
            idx_pos = c*mX + idx[i]-1; //subtract 1 to account for
            idx_neg = c*mX + idx[i+nIdx]-1; //matlab style indexing
            
            Qnew = (X[idx_pos]-X[idx_neg])/(2*wgt[i]) + Q[idx_Q];
            Qnew = sgn(Qnew)*min(fabs(Qnew),lam);
            
            X[idx_pos] = X[idx_pos]+wgt[i]*(Q[idx_Q]-Qnew);
            X[idx_neg] = X[idx_neg]-wgt[i]*(Q[idx_Q]-Qnew);
            
            Q[idx_Q] = Qnew;
        }
    }
    
}

void UpdateTVDual(double X[], double Q[], const int idx[], 
        const double wgt[], const double lam[], const int mX, const int nX, 
        const int nIdx) {
    
    for(int i=0; i<nX; i++)
        UpdateTVDual(&X[i*mX],&Q[i*nIdx],idx,wgt,lam[i],mX,1,nIdx);
}

void Update0DiagDual(double X[], double Q[], const double V[],
        const double V2[], const int mX, const int nX, const int nV) {
    
    double deltaQ = 0;
    
    for(int i=0; i<mX; i++) {
        if(V2[i]>0) {
            deltaQ = 0;
            
            for(int r=0; r<nV; r++) 
                deltaQ += X[mX*r+i]*V[mX*r+i];
            
            deltaQ = deltaQ/V2[i];
            
            Q[i] += deltaQ;
            
            for(int r=0; r<nV; r++)
                X[mX*r+i] = X[mX*r+i]-deltaQ*V[mX*r+i];
        }
        else;
    }
}

void UpdateSum1Dual(double X[], double Q[], const proxBOOL sum_dir,
        const int mX, const int nX) {
    
    double sumX = 0;
    double deltaQ = 0;
    
    if(sum_dir == COL_SUM) {
        //here we're constraining the column sums
        
        for(int j=0; j<nX; j++) {
            sumX = 0;
            
            for(int i=0; i<mX; i++)
                sumX += X[j*mX+i];
            
            deltaQ = (sumX-1)/((double)mX);
            
            for(int i=0; i<mX; i++)
                X[j*mX+i] -= deltaQ;
            
            Q[j] += deltaQ;
        }
    }
    else {
        //here we're constraining the row sums
        
        for(int i=0; i<mX; i++) {
            sumX = 0;
            
            for(int j=0; j<nX; j++)
                sumX += X[j*mX+i];
            
            deltaQ = (sumX-1)/((double)nX);
            
            for(int j=0; j<nX; j++)
                X[j*mX+i] -= deltaQ;
            
            Q[i] += deltaQ;
        }
    }
}

void InitTV(double X[], double Qout[], const double Qinit[],
        const int idx[], const double wgt[], const double lam,
        const int mX, const int nX, const int nIdx) {
    
    double Qnew = 0;
    int idx_pos = 0;
    int idx_neg = 0;
    int idx_Q = 0;
    
    for(int c=0; c<nX; c++) {
        for(int i=0; i<nIdx; i++) {
            idx_Q = c*nIdx + i;
            idx_pos = c*mX + idx[i]-1; //subtract 1 to account for
            idx_neg = c*mX + idx[i+nIdx]-1; //matlab style indexing
            
            Qnew = Qinit[idx_Q];
            Qnew = sgn(Qnew)*min(fabs(Qnew),lam);
            
            X[idx_pos] = X[idx_pos]-wgt[i]*Qnew;
            X[idx_neg] = X[idx_neg]+wgt[i]*Qnew;
            
            Qout[idx_Q] = Qnew;
        }
    }
}


void InitTV(double X[], double Qout[], const double Qinit[], 
        const int idx[], const double wgt[], const double lam[], 
        const int mX, const int nX, const int nIdx) {
    
    for(int i=0; i<nX; i++)
        InitTV(&X[i*mX],&Qout[i*nIdx],&Qinit[i*nIdx],idx,wgt,lam[i],mX,1,nIdx);
}

double CalculateFro2Norm(const double X[], const int length) {
    double out = 0;
    
    for(int i=0; i<length; i++)
        out += X[i]*X[i];
    
    return out;
}

double CalculateTVNorm(const double X[], const int idx[], const double wgt[],
        const int mX, const int nX, const int nIdx) {
    double out = 0;
    
    int idx_pos = 0;
    int idx_neg = 0;
    
    for(int c=0; c<nX; c++) {
        for(int i=0; i<nIdx; i++) {
            idx_pos = c*mX + idx[i]-1; //subtract 1 for matlab indexing
            idx_neg = c*mX + idx[i+nIdx]-1;
            
            out += wgt[i]*fabs(X[idx_pos]-X[idx_neg]);
        }
    }
    
    return out;
}

double CalculateL1Norm(const double X[], const int length) {
    double out = 0;
    
    for(int i=0; i<length; i++)
        out += fabs(X[i]);
    
    return out;
}

double DotProd(const double Arr1[], const double Arr2[],const int length) {
    double out = 0;
    
    for(int i=0; i<length; i++)
        out += Arr1[i]*Arr2[i];
    
    return out;
}

void * CreateVar(mwSize n, mwSize size, void * vars[], int &num_vars) {
    
    void * out;
    
    if(num_vars+1 < MAX_NUM_VAR) {
        out = mxCalloc(n,size);
        vars[num_vars] = out;
        num_vars += 1;
        
        return out;
    }
    else
        mexErrMsgTxt("Maximum variable number exceeded");
}

void CleanupMem(void * vars[], const int num_vars) {
    
    for(int i=0; i<num_vars; i++)
        mxFree(vars[i]);
}