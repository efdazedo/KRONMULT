#ifndef KRONMULT2_BATCHED_HPP
#define KRONMULT2_BATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult2.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) = kron(A1(k),A2(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult2_batched(
                       int const n,
                       T const Aarray_[],
                       T X_[],
                       T Y_[],
                       T W_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (n,n,2,batchCount)
// X_ is (n^2, batchCount)
// Y_ is (n^2, batchCount)
// W_ is (n^2, batchCount)
//
{
#ifdef USE_GPU
        // -------------------------------------------
        // note 1-based matlab convention for indexing
        // -------------------------------------------
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
#else
        int const iz_start = 1;
        int const iz_size = 1;
#endif

        int const n2 = n*n;


#define X(i,j) X_[ indx2f(i,j,n2) ]
#define Y(i,j) Y_[ indx2f(i,j,n2) ]
#define W(i,j) W_[ indx2f(i,j,n2) ]
#define Aarray(i1,i2,i3,i4) Aarray_[ indx4f(i1,i2,i3,i4, n,n,2 ) ]

        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp = &( X(1,ibatch) );
                T* const Yp = &( Y(1,ibatch) );
                T* const Wp = &( W(1,ibatch) );
                T const * const A1 = &(Aarray(1,1,1,ibatch));
                T const * const A2 = &(Aarray(1,1,2,ibatch));
                int const nvec = 1;
                kronmult2( n, nvec, A1,A2, Xp, Yp, Wp );
        };

}

#undef X
#undef Y
#undef W
#undef Aarray


                       



#endif
