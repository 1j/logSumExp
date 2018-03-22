#ifndef LOGSUM_HPP_
#define LOGSUM_HPP_

#include <math.h>
#include "vector/vectorclass.h"
#include "vector/vectormath_exp.h"
#include <vector>
#include <limits>


#ifdef __SSE__
	#ifdef __AVX__ // using 256-bit FP instructions 
		template <typename T>
		struct vec {
			static const int width = 32 / sizeof(T);
			typedef typename std::conditional<std::is_same<T,float>::value, Vec8f, Vec4d>::type v;
		};
	#else // using 128-bit SSE2
		template <typename T>
		struct vec {
			static const int width = 16 / sizeof(T);
			typedef typename std::conditional<std::is_same<T,float>::value, Vec4f, Vec2d>::type v;
		};
	#endif

	template <int N> struct _int{ };

	template <int N, typename T, typename V>
	inline void unrolled_load(T& vn, V& logV, int i, int w, _int<N>) {
		unrolled_load(vn, logV, i, w, _int<N-1>());
		vn[N-1].load(logV+i+(N-1)*w);
	}
	template <typename T, typename V>
	inline void unrolled_load(T& vn, V& logV, int i, int w, _int<1>) {
		vn[0].load(logV+i);
	}

	template <int N, typename T>
	inline void unrolled_exp_add(T& an, T& vn, _int<N>) {
		unrolled_exp_add(an, vn, _int<N-1>());
		an[N-1] += exp(vn[N-1]);
	}
	template <typename T>
	inline void unrolled_exp_add(T& an, T& vn, _int<1>) {
		an[0] += exp(vn[0]);
	}

	template <int N, typename T>
	inline void unrolled_subtract(T* vn, T& mm, _int<N>) {
		unrolled_subtract(vn, mm, _int<N-1>());
		vn[N-1] -= mm;
	}
	template <typename T>
	inline void unrolled_subtract(T* vn, T& mm, _int<1>) {
		vn[0] -= mm;
	}

        template <int N, typename T>
	inline void unrolled_max(T& vn, T& mn, _int<N>) {
		unrolled_max(vn, mn, _int<N-1>());
		mn[N-1] = max(mn[N-1],vn[N-1]);
	}
	template <typename T>
	inline void unrolled_max(T& vn, T& mn, _int<1>) {
		mn[0] = max(mn[0],vn[0]);
	}

	/**
	 * Finds the maximum element of an array using SIMD instructions.
	 * @tparam N Number of accumulators (usually 3 is best)
	 * @tparam T Type of contents of array (should be floating point)
	 * @param v The input array
	 * @param length The length of the array
	 */
	template <typename T, int N>
	inline T max_element(const T* v, int length) {
		if (length==1) return v[0];
		int w = vec<T>::width;
		if (length < N*w) {
			T m = v[0];
			for (int i=1; i<length; i++) m = fmax(m,v[i]);
			return m;
		}
		int l = length - length % (w*N);
		typedef typename vec<T>::v vv;
		vv vn[N], mn[N];
		for (int n=0; n<N; n++) {
			mn[n] = vv().load(v+n*w);
		}
		for (int i=N*w; i<l; i+=N*w) {
			unrolled_load(vn,v,i,w,_int<N>() );
			unrolled_max(vn,mn,_int<N>());
		}
		for (int n=1; n<N; n++) {
			mn[0] = max(mn[0],mn[n]);
		}

		T m = mn[0][0];
		for (int i=1; i<w; i++) m = fmax(m,mn[0][i]); // Not particularly efficient
		if (l < length) for (int i=l; i<length; i++) m = fmax(m,v[i]);
		return m;
	}

	/**
	 * A numerically stable way of computing
	 *
	 *  logSum(logV,n) =  log(sum(exp(logV)))
	 *
	 * using SIMD instructions.
	 *
	 * @tparam N Number of accumulators
	 * @tparam T Type of contents of array (should be floating point)
	 * @param logV The input array
	 * @param length The length of the array
	 */
	template <typename T, int N>
	inline T logSum(const T* logV, int length) {
		T m = max_element<T,N>(logV,length);
		int w = vec<T>::width;
		int l = length - length % (w*N);
		typedef typename vec<T>::v vv;
		vv vn[N], an[N], mm(m);
		for (int n=0; n<N; n++) an[n] = vv(0.0);
     	        if (l >= N*w) {
			for (int i=0; i<l; i+=N*w) {
				unrolled_load(vn,logV,i,w,_int<N>() );
				unrolled_subtract(vn,mm,_int<N>());
				unrolled_exp_add(an,vn,_int<N>());
			}
     	        }
		// Finish off the end of the vector if needed
		T s = 0;
		if (length < N*w || l < length) {
			for (int i=l; i<length; i++) {
				s += exp(logV[i]-m);
			}
		}
		if (length >= N*w) {
			for (int n=1; n<N; n++) an[0] += an[n];
			return (mm + log(vv(s+horizontal_add(an[0]))))[0];
		}
		else return m + log(s); // TODO Coerce this to use non-IEEE log
	}

	/**
	 * A numerically stable way of computing
	 *
	 *  logSum(logV,n) =  log(sum(exp(logV)))
	 *
	 * using SIMD instructions.
	 *
	 * @tparam T Type of contents of array (should be floating point)
	 * @param logV The input array
	 * @param length The length of the array
	 */
        template <typename T>
        inline T logSum(const T* logV, int length) {
	  //  default here is to use 8 accumulators
	  return logSum<T,8>(logV,length);
	}

       /**
        * Wrapper functions to allow for runtime selection of the number of accumulators.
        *
        * @tparam N The maximum number of accumulators
        * @param logV The input array
        * @param length The length of the array
        * @param n The number of accumulators specified at runtime
        */
        template <typename T, int N>
	inline double logSumN(const T* logV, int length, int n, _int<N>) {
	  if (n==N) return logSum<T,N>(logV,length);
	  else return logSumN(logV,length,n,_int<N-1>());
	}
        // Fall back to default if n was not in the range [1,N] 
        template <typename T>
        inline double logSumN(const T* logV, int length, int n, _int<1>) {
	  return logSum(logV,length);
	}

        template <typename T>
	inline void logAdd(T* logV1, const T* logV2, int length) {
	  int w = vec<T>::width;
	  int l = length - length % w;
	  typedef typename vec<T>::v vv;
	  vv ma, mi, v1, v2;
	  if (l >= w) {
		  for (int i=0; i<l; i+=w) {
			  v1.load(logV1+i);
			  v2.load(logV2+i);
			  ma = max(v1,v2);
			  mi = min(v1,v2);
			  ma += log1p(exp(mi - ma));
			  ma.store(logV1+i);
		  }
	  }
	  // Finish off the end of the vector if needed
	  T mma, mmi;
	  if (length < w || l < length) {
		  for (int i=l; i<length; i++) {
			  mma = fmax(logV1[i],logV2[i]);
			  mmi = fmin(logV1[i],logV2[i]);
			  logV1[i] = mma + log1p(exp(mmi - mma));
		  }
	  }
	}
#else
// fall back to non-SIMD version
	template <class T>
	inline double max_element(const T* v, int length) {
		if (length==1) return v[0];
		T max = v[0];
		for (int i=1; i<length; i++) max = fmax(max,v[i]);
		return max;
	}
        template <class T, int N>
	inline double logSum(T* logV, int length) {
	  T m = max_element(logV,length);
	  T expSum = 0;
	  for (int i=0; i<length; i++) expSum += exp(logV[i]-m);
	  return m + log(expSum);
	}
	template <typename T>
	inline void logAdd(T* logV1, const T* logV2, int length) {
	    for (int i=0; i<length; i++) {
	      double big = logV1[i];
	      double small = logV2[i];
	      if (big < small) {
		std::swap(big,small);
	      }      
	      logV1[i] = big + log1p(exp(small - big));
	    }
	}

#endif

#endif /* LOGSUM_HPP_ */
