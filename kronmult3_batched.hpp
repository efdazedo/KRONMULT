#ifndef KRONMULT3_BATCHED_HPP
#define KRONMULT3_BATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmult3.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) = kron(A1(k),...,A3(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T>
GLOBAL_FUNCTION
void kronmult3_batched(
                       int const n,
                       T const Aarray_[],
                       T X_[],
                       T Y_[],
                       T W_[],
                       int const batchCount)
//
// conceptual shape of Aarray is  (n,n,3,batchCount)
// X_ is (n^3, batchCount)
// Y_ is (n^3, batchCount)
// W_ is (n^3, batchCount)
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
        int const n3 = n*n2;


#define X(i,j) X_[ indx2f(i,j,n3) ]
#define Y(i,j) Y_[ indx2f(i,j,n3) ]
#define W(i,j) W_[ indx2f(i,j,n3) ]
#define Aarray(i1,i2,i3,i4) Aarray_[ indx4f(i1,i2,i3,i4, n,n,3 ) ]

        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T* const Xp = &( X(1,ibatch) );
                T* const Yp = &( Y(1,ibatch) );
                T* const Wp = &( W(1,ibatch) );
                T const * const A1 = &(Aarray(1,1,1,ibatch));
                T const * const A2 = &(Aarray(1,1,2,ibatch));
                T const * const A3 = &(Aarray(1,1,3,ibatch));
                int const nvec = 1;
                kronmult3( n, nvec, A1,A2,A3, Xp, Yp, Wp );
        };

}

#undef X
#undef Y
#undef W
#undef Aarray


                       



#endif
