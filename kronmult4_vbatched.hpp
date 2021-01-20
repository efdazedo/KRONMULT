#ifndef KRONMULT4_VBATCHED_HPP
#define KRONMULT4_VBATCHED_HPP 1


#include "kronmult_vbatched.hpp"




// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A4(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult4_vbatched(
                       int const m[],
		       int const n[],
                       T const * const Aarray_[],
                       T* pX_[],
                       T* pY_[],
                       T* W_,
		       size_t const Wcapcity,
                       int const batchCount
		       )
//
// conceptual shape of Aarray(ibatch) is  (m(ibatch), n(ibatch) )
//
// pX_[] is array of pointers to X[], each of size prod(n(1:ndim))
// pY_[] is array of pointers to Y[], each of size prod(m(1:ndim))
//
// W_[] is array of  size Wcapcity
//
// Y is the output
// X is the input (but may be modified)
// W is workspace
//
//
{
        int constexpr ndim = 4;
	kronmult_vbatched<T,ndim>(
			m, n, Aarray_, pX_, pY_, W_, Wcapcity,batchCount );
}



#endif
