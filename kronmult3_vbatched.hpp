#ifndef KRONMULT3_VBATCHED_HPP
#define KRONMULT3_VBATCHED_HPP 1


#include "kronmult_vbatched.hpp"




// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A3(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult3_vbatched(
		       int const m1, int const n1,
		       int const m2, int const n2,
		       int const m3, int const n3,
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
        int constexpr ndim = 3;
	int const m4 = 1; int const n4 = 1;
	int const m5 = 1; int const n5 = 1;
	int const m6 = 1; int const n6 = 1;

	kronmult_vbatched<T,ndim>(
			m1, n1, 
			m2, n2, 
			m3, n3, 
			m4, n4, 
			m5, n5, 
			m6, n6, 
			Aarray_, pX_, pY_, W_, Wcapcity,batchCount );
}



#endif
