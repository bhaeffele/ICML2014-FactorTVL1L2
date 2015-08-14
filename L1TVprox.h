//Header file for various functions related to minimizing L1 and total 
//variation style proximal operators with various constraints.
//
//Ben Haeffele - Oct 2013

typedef bool proxBOOL;

#define ROW_SUM true
#define COL_SUM false

#define MAX_NUM_VAR 100 //Maximum number of pointers to variables that
                             //we'll allow to be created.

#ifndef MEX
#define MEX
#include "mex.h"
#endif

#include "math.h"

//templates for the max and min functions (for some reason they're not in
//the standard library I have on my machine
template <typename T> T max(T a, T b) {
    return (a<b)? b : a;
}

template <typename T> T min(T a, T b) {
    return (a<b)? a : b;
}

//template to calculate the sign function
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

//template for the soft threshold function
template <typename T> T softthresh(T val,T thresh) {
    return sgn(val)*max(fabs(val)-thresh,T(0));
}

//teplate for non-negative threshold function
template <typename T> T posthresh(T val, T thresh) {
    return (val > thresh)? val-thresh : 0;
}

//Soft thresholding on an entire array.
//
//Y[i] = {0 if abs(X[i])<=lam, X[i]-lam if X[i]>lam, X[i]+lam if X[i]<-lam
//
//Inputs:
//X - Input array
//lam - thresholding parameter
//length - number of elements in array
//
//Outputs:
//Y - Output array
void SoftThresholdMatrix(const double X[], double Y[], const double lam, 
        const int length);

//Soft thresholding of an array with different regularization parameters
//for each column of the matrix
//
//Inputs:
//X - Input array
//lam - Vector of threshold parameters (should have length equal to nX)
//mX - number of rows in X
//nX - number of columns in X
void SoftThresholdMatrix(const double X[], double Y[], const double lam[],
        const int mX, const int nX);

//Positive soft thresholding on an entire array.
//
//Y[i] = {0 if X[i]<=lam, X[i]-lam otherwise
//
//Inputs:
//X - Input array
//lam - thresholding parameter
//length - number of elements in array
//
//Outputs;
//Y - Output array
void PosSoftThresholdMatrix(const double X[], double Y[], const double lam, 
        const int length);

//Positive soft thresholding of an array with different regularization
//parameters for each column of the matrix
//
//Inputs:
//X - Input array
//lam - Vector of threshold parameters (should have length equal to nX)
//mX - number of rows in X
//nX - number of columns in X
void PosSoftThresholdMatrix(const double X[], double Y[], 
        const double lam[], const int mX, const int nX);

//Update L1 dual variables
//
//The update for L1 dual variables is given as:
//
// min_{Q} 0.5*|X-Q|_F^2 s.t. abs(Q)<=lam
//
//Inputs:
//X - Residual to fit (this gets changed in the function to reflect the 
//      updated dual variables)
//Q - Array of dual variables (should be same size as X)
//lam - constraint parameter
//length - number of elements in X (and Q)
void UpdateL1Dual(double X[], double Q[], const double lam, 
        const int length);

//Update L1 dual variables with different regularization parameters for 
//each column of Q
//
//The update for a column of L1 dual variables is given as:
//
// min_{Qi} 0.5*|X-Qi|_F^2 s.t. abs(Qi)<=lam(i)
//
//Inputs:
//X - Residual to fit (this gets changed in the function to reflect the 
//      updated dual variables)
//Q - Array of dual variables (should be same size as X)
//lam - Vector of constraint parameters (should have length = number of 
//          columns in X)
//mX - number of rows in X
//nX - numbre of columns in X
void UpdateL1Dual(double X[], double Q[], const double lam[],
        const int mX, const int nX);

//Update total variation dual variables
//
//The update of the variables is given as:
//
// min_(Q) 0.5*|X-G'*Q|_F^2 s.t. abs(Q)<=lam
//
//where G is a matrix that calculates the weighted difference between
//connected elements in the columns of Q.  Because G is often very large 
//and sparse we instead just pass an array containing the indices to take 
//the difference between and a vector of weights.
//
//Inputs:
//X - Residual to fit (this gets changed during the function to reflect the 
//      updated dual variables)
//Q - Array of dual variables (should be of size nIdx x nX)
//idx - nIdx x 2 array containing the neighbor pair indices (note that it 
//       is assumed these are 1 index (Matlab style)
//wgt - nIdx x 1 vector containing the weight to apply to each pair
//lam - constraint parameter
//mX - number of rows in X
//nX - number of columns in X
//nIdx - number of rows in idx
void UpdateTVDual(double X[], double Q[], const int idx[], 
        const double wgt[], const double lam, const int mX, const int nX, 
        const int nIdx);

//Update total variation dual variables with a different regularization
//parameter for each column of Q
//
//The update of the variables is given as:
//
// min_(Qi) 0.5*|X-G'*Qi|_F^2 s.t. abs(Qi)<=lam(i)
//
//where G is a matrix that calculates the weighted difference between
//connected elements in the columns of Q.  Because G is often very large 
//and sparse we instead just pass an array containing the indices to take 
//the difference between and a vector of weights.
//
//Inputs:
//X - Residual to fit (this gets changed during the function to reflect the 
//      updated dual variables)
//Q - Array of dual variables (should be of size nIdx x nX)
//idx - nIdx x 2 array containing the neighbor pair indices (note that it 
//       is assumed these are 1 index (Matlab style)
//wgt - nIdx x 1 vector containing the weight to apply to each pair
//lam - Vector of constrain parameters (should have length nX)
//mX - number of rows in X
//nX - number of columns in X
//nIdx - number of rows in idx
void UpdateTVDual(double X[], double Q[], const int idx[], 
        const double wgt[], const double lam[], const int mX, const int nX, 
        const int nIdx);

//Update dual variables that enforce diagonal being 0
//
//For the constraint that diag(X*V') = 0
//
//Dual variable updates are given as:
//
// Q(i) = Q(i) + X(i,:)*Q(i,:)'/(Q(i,:)*Q(i,:)')
//
//Here we assume that the Fro^2 norms of the rows of Q has already been
//computed and require it as an input parameter.
//
//Inputs:
//X - Residual to fit (this gets changed in the function to reflect the
//      updated dual variables) (size mX x nX)
//Q - Array of dual variables (size mX x nV)
//V - Array in zero diagonal constraint (size mX x nV)
//V2 - Vector containing the Fro^2 norms of the rows of V (size mX x 1)
//mX - number of rows in X
//nX - number of columns in X
//nV - number of columns in V
void Update0DiagDual(double X[], double Q[], const double V[], 
        const double V2[], const int mX, const int nX, const int nV);

//Update dual variables that enforce either the row or column sum being 0.
//
//For a constraint on the column sums, sum(X,1) == ones(1,size(X,2))
//
//The dual variable updates are given as:
//
// Q(i) = Q(i) + (sum(X(:,i))-1)/size(X,1)
//
//For a constraint on the row sums, sum(X,2) == ones(size(X,1),1)
//
//The dual variable updates are given as:
//
// Q(i) = Q(i) + (sum(X(i,:))-1)/size(X,2)
//
//Inputs:
//X - Residual to fit (this gets changed in the function to reflect the
//      updated dual variables) (size mX x nX)
//Q - Vector of dual variables with length mX or nX for a row sum or column
//      sum constraint, respectively.
//sum_dir - Boolean definining whether the constraint is on row sums or 
//          column sums.  If sum_dir==COL_SUM then the column sums are
//          constrained to sum to 1.  If sum_dir==ROW_SUM then the row sums
//          are constrained to sum to 1.
//mX - number of rows in X
//nX - number of columns in X
void UpdateSum1Dual(double X[], double Q[], const proxBOOL sum_dir,
        const int mX, const int nX);

//Function to initialize the Qtv dual variables.
//
//Inputs:
//X - Starting residual (this gets changed in the function to reflect the 
//      updated dual variables) (size mX x nX)
//Qout - Out array of dual variables (size nIdx x nX)
//Qinit - Array of initializations for the dual variables (size nIdx x nX)
//idx - nIdx x 2 array containing the neighbor pair indices (note that it 
//       is assumed these are 1 index (Matlab style)
//wgt - nIdx x 1 vector containing the weight to apply to each pair
//lam - Constraint parameter
//mX - number of rows in X
//nX - number of columns in X
//nIdx - number of rows in idx
void InitTV(double X[], double Qout[], const double Qinit[], 
        const int idx[], const double wgt[], const double lam, 
        const int mX, const int nX, const int nIdx);

//Function to initialize the Qtv dual variables.  Overloaded function of 
//that given above.  Only difference is now lam can be a vector with
//different values for each column of Qtv.
//
//lam - Vector of constraint parameters (length nX)
void InitTV(double X[], double Qout[], const double Qinit[], 
        const int idx[], const double wgt[], const double lam[], 
        const int mX, const int nX, const int nIdx);

//Calculate squared Frobenius norm of X
//
//Inputs:
//X - Input matrix
//length - Number of elements in X
//
//Returns:
//Sum of the square of every element in X
double CalculateFro2Norm(const double X[], const int length);

//Calculate TV norm of X
//
//See UpdateTVDual function for a description of the input parameters.
//
//Returns:
//TV norm of X
double CalculateTVNorm(const double X[], const int idx[], const double wgt[],
        const int mX, const int nX, const int nIdx);

//Calculate L1 norm of X
//
//Inputs:
//X - Input matrix
//length - Number of elements in X
//
//Returns:
//L1 norm of X
double CalculateL1Norm(const double X[], const int length);

//DotProd
//
//Dot product of two arrays
//
//Inputs:
//Arr1 - Input array 1
//Arr2 - Input array 2
//length - Length of arrays
//
//Returns:
//Dot product of the two arrays
double DotProd(const double Arr1[],const double Arr2[],const int length);

//Allocate memory for a variable and register the pointer with our list of
//variables
//
//Inputs:
//n - Number of elements in new array
//size - Size of each element in new array
//vars - Pointer array of all of the variables we have created
//num_vars - Number of variables we have created (modified in function)
//
//Returns:
//Pointer to newly created array
void * CreateVar(mwSize n, mwSize size, void * vars[], int &num_vars);

//Function to free all of the memory that we have used
//
//Inputs:
//vars - Pointer array of all of the variables we have created
//num_vars - Number of variables that have been created
void CleanupMem(void * vars[], const int num_vars);