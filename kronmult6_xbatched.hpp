#ifndef KRONMULT6_XBATCHED_HPP
#define KRONMULT6_XBATCHED_HPP 1


#include "kronmult_xbatched.hpp"




// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A6(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult6_xbatched(
                       int const n,
                       T const * const Aarray_[],
		       int const lda,
                       T* pX_[],
                       T* pY_[],
                       T* pW_[],
                       int const batchCount,
		       int const subbatchCount = 0)
//
// conceptual shape of Aarray is  (ndim,batchCount)
//
// pX_[] is array of pointers to X[], each of size n^6
// pY_[] is array of pointers to Y[], each of size n^6
// pW_[] is array of pointers to Z[], each of size n^6
//
// Y is the output
// X is the input (but may be modified)
// W is workspace
//
//
{
	kronmult_xbatched<T,6>(
			n, Aarray_, lda, pX_, pY_, pW_, batchCount, subbatchCount );
}



#endif
