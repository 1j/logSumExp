#include <Rcpp.h>
#include <math.h>
#include "logSum.hpp"
using namespace Rcpp;

//' Stable computation of the logarithm of a sum of values
//'
//' Numerically stable computation of
//'   \code{log(sum(exp(logV)))}
//' using SIMD instructions.
//' @param logV The input vector.
//' @param accumulators The number of accumulators used (maximum 12).
//' Typically 8 is fastest, but this may be architecture-dependent.
//' For values in the range [2,12], the difference in speed is typically
//' not large (less than 10%).
//' @examples
//' logx = runif(1e5,-10000,-9000)
//' log(sum(exp(logx)))
//' ## [1] -Inf
//' logSumExp(logx)
//' ## [1] -8995.453
//'
//' ## Example with a normalized vector
//' x = runif(1e5)
//' x = x/sum(x)
//' logSumExp::logSumExp(log(x))
//' ## [1] 0
//' @details See \url{https://github.com/1j/logSumExp}
// [[Rcpp::export]]
double logSumExp (NumericVector& logV, int accumulators=8) {
  constexpr int MAX_ACCUMULATORS = 12;
  if (accumulators>MAX_ACCUMULATORS) accumulators = MAX_ACCUMULATORS;  
  return logSumN(&logV[0],logV.size(),accumulators,_int<MAX_ACCUMULATORS>());
}

//' Stable computation of the logarithm of a sum for columns of a matrix
//'
//' Numerically stable computation of
//'   \code{apply(X,2,function(x) log(sum(exp(x))))}
//' using SIMD instructions. Note that no function \code{rowLogSumExps}
//' is provided. If this is desired, the matrix should first be transposed
//' and then can be passed into \code{colLogSumExps}.
//' @param logV The input vector.
//' @param accumulators The number of accumulators used (maximum 12).
//' Typically 5 is fastest here, but this may be architecture-dependent.
//' For values in the range [2,12], the difference in speed is typically
//' not large (less than 10%).
//' @examples
//' logx = matrix(runif(1e6,-10000,-9000),1000,1000)
//' logsum = colLogSumExps(logx)
//' @details See \url{https://github.com/1j/logSumExp}
// [[Rcpp::export]]
NumericVector colLogSumExps(NumericMatrix& logV, int accumulators=5) {
  constexpr int MAX_ACCUMULATORS = 12;
  if (accumulators>MAX_ACCUMULATORS) accumulators = MAX_ACCUMULATORS;    
  int nCol = logV.ncol();
  int nRow = logV.nrow();
  NumericVector result(nCol);
  for (int j=0; j<nCol; j++) {
    result[j] = logSumN(&logV[0] + j*nRow,nRow,accumulators,_int<MAX_ACCUMULATORS>());
  }
  return(result);
}

//' Stable computation of element-wise sum of two vectors in log space
//'
//' Numerically stable computation of
//'   \code{log(exp(logA)+exp(logB))}
//' using SIMD instructions. 
//' @param logA The first input vector.
//' @param logB The second input vector.
//' @examples
//' ## create two vectors of similarly small log values
//' logA = runif(1e5,-10000,-9000)
//' logB = logA + rnorm(1e5,0,1)
//'
//' ## compute sum in log space
//' logC = logAddExp(logA,logB)
//' head(cbind(logA,logB,logC))
//' ##           logA      logB      logC
//' ## [1,] -9782.350 -9781.286 -9780.990
//' ## [2,] -9279.501 -9279.381 -9278.746
//' ## [3,] -9402.641 -9402.887 -9402.064
//' ## [4,] -9927.700 -9926.675 -9926.369
//' ## [5,] -9586.167 -9586.246 -9585.513
//' ## [6,] -9602.263 -9603.234 -9601.942
//' @details See \url{https://github.com/1j/logSumExp}
// [[Rcpp::export]]
NumericVector logAddExp(NumericVector& logA,NumericVector& logB) {

  NumericVector result;
  if (logA.size() != logB.size()) {
    std::cout << "logAddExp: Vectors must be of same length.\n";
    result = NumericVector(1);
    result[0] = NA_REAL;
  }
  else {
    result = clone(logA);
    logAdd(&result[0],&logB[0],logA.size());
  }
  return result;
}
