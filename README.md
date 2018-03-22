# logSumExp
This package provides functionality for numerically stable computation of the log of a sum of very small values, using SIMD instructions. This is required for many types of probabilitistic inference, for example when normalizing a probability distribution, where each individual outcome may have a very low probability (e.g. in forward-backward algorithms for HMMs).

With AVX instructions, this can be an order of magnitude faster than serial implementations (e.g. `matrixStats::logSumExp`). The code makes use of the `C++` vector class library from Agner Fog (http://www.agner.org/optimize/#vectorclass).

# Installation

The code is written in `C++`, wrapped in an `R` package. To install the `R` package, use
```r
devtools::install_github("1j/logSumExp")
```

# Examples

Create a long vector of large negative values, representing the log of very small numbers (these may represent log likelihoods, for example):
```r
lx = runif(1e5,-10000,-9000)
```
Naive computation of the log of the sum of the exponentiated values results in numerical underflow:
```r
log(sum(exp(lx)))
## [1] -Inf
```
Stable algorithm for computing the result:
```r
matrixStats::logSumExp(lx)
## [1] -8995.453
logSumExp::logSumExp(lx)
## [1] -8995.453
```

# Microbenchmark results

All benchmarks carried out on a quad-core 2.30GHz Intel i5-6200U machine.

## logSumExp

```r
microbenchmark(matrixStats::logSumExp(lx),times=100)
## Unit: milliseconds
##                        expr      min       lq     mean   median       uq      max neval
##  matrixStats::logSumExp(lx) 2.635924 2.643336 2.670215 2.646272 2.662407 3.648855   100

## On a CPU with SSE and no AVX
microbenchmark(logSumExp::logSumExp(lx),times=100)
## Unit: microseconds
##          expr     min       lq     mean  median      uq     max neval
## logSumExp(lx) 601.214 602.9115 615.9532 607.214 610.091 850.491   100

## On a CPU with AVX
microbenchmark(logSumExp::logSumExp(lx),times=100)
## Unit: microseconds
##                      expr     min      lq     mean  median       uq     max neval
##  logSumExp::logSumExp(lx) 285.076 286.099 290.9375 287.352 290.4425 430.876   100
```

## colLogSumExps

Simply using `apply(X,2,logSumExp)` already gives a speedup of around 30% over the serial implementation
```r
lxx = matrix(runif(1e6,-10000,-9000),1000,1000)

microbenchmark(matrixStats::colLogSumExps(lxx),times=100)
## Unit: milliseconds
##                            expr      min       lq     mean   median       uq      max neval
## matrixStats::colLogSumExps(lxx) 26.36714 26.49099 26.67565 26.57162 26.73154	30.78032   100

microbenchmark(apply(lxx,2,logSumExp::logSumExp),times=100)
## Unit: milliseconds
##                                     expr      min       lq     mean   median       uq     max neval
##  apply(lxx, 2, function(x) logSumExp(x)) 17.26169 18.90467 47.62767 20.72705 98.35533 192.129   100
```
We can do better than this by using the function `colLogSumExps`, which makes the individual calls to `logSumExp` from within the `C++` code, resulting in a further 6-fold speedup.
```r
microbenchmark(logSumExp::colLogSumExps(lxx,accumulators=5),times=100)
## Unit: milliseconds
                                            expr      min       lq     mean
 logSumExp::colLogSumExps(lxx, accumulators = 5) 3.589024 3.630719 3.668657
   median       uq      max neval
 3.645854 3.687575 4.209389   100
```

# Further details

## Numerical precision
```r
x = runif(1e5)
x = x/sum(x)
matrixStats::logSumExp(log(x))
## [1] -5.329071e-15
logSumExp::logSumExp(log(x))
## [1] 0
x[1] = x[1] + 3e-14
log(sum(x))
## [1] 2.997602e-14
logSumExp::logSumExp(log(x))
## [1] 3.019807e-14
matrixStats::logSumExp(log(x))
## [1] 2.131628e-14
```

## Handling of special cases (`NA`, `Inf`, `NaN`)
For efficiency reasons, the `R` wrapper code does not explicitly check for special cases. Vectors containing these values will be treated in the following manner:
```r
logSumExp(c(-Inf,1))
## [1] 1
logSumExp(c(-Inf,-Inf))
## [1] NaN
logSumExp(c(Inf,1))
## [1] NaN
logSumExp(c(NA,1))
## [1] NA
logSumExp(c(NaN,1))
## [1] NaN
logSumExp(c(NaN,NA))
## [1] NaN
```
## NA removal

```r
x = c(1:10,NA)
matrixStats::logSumExp(log(x))
## [1] NA
matrixStats::logSumExp(log(x),na.rm=TRUE)
## [1] 4.007333
logSumExp::logSumExp(log(x))
## [1] NA
logSumExp::logSumExp(log(x[!is.na(x)]))
## [1] 4.007333
```
Even naively removing the `NA` values upon every call to `logSumExp` still works out almost 50% faster than the `matrixStats` version, although the optimal approach will be to filter out `NA` values outside of any loops wherever possible.
```r
microbenchmark(logSumExp::logSumExp(x[!is.na(x)]),times=100)
## Unit: microseconds
##                                expr  min      lq     mean  median      uq     max neval
##  logSumExp::logSumExp(x[!is.na(x)]) 89.4 93.6395 100.2074 94.5345 96.8425 163.069   100
## microbenchmark(matrixStats::logSumExp(x,na.rm=TRUE),times=100)
## Unit: microseconds
##                                     expr     min       lq     mean   median       uq     max	neval
##  matrixStats::logSumExp(x, na.rm = TRUE) 166.662 169.3715 173.3236 170.0925 172.5585 231.021	100 
```

