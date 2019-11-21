
#ifndef KGEMM_NT_BATCHED_H
#define KGEMM_NT_BATCHED_H 1

#include "kgemm_nt.hpp"

#ifdef USE_GPU
#define GLOBAL __global__
#else
#define GLOBAL
#endif

template<typename T>
GLOBAL_FUNCTION
void kgemm_nt_batched( int const mm, int const nn, int const kk, 
                       T const alpha, 
                       T* const Aarray_[], 
                       int const ldAarray_[], 
                       T* const Barray_[], 
                       int const ldBarray_[], 
                       T const beta,  
                       T* const Carray_[], 
                       int const ldCarray_[], 
                       int const batchCount)
{
// ----------------------------
// use Fortran 1-based indexing
// ----------------------------
#define Aarray(i)  Aarray_[ (i) - 1]
#define Barray(i)  Barray_[ (i) - 1]
#define Carray(i)  Carray_[ (i) - 1]

#define ldAarray(i) ldAarray_[ (i) - 1]
#define ldBarray(i) ldBarray_[ (i) - 1]
#define ldCarray(i) ldCarray_[ (i) - 1]


#ifdef USE_GPU
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
#else
        int const iz_start = 1;
        int const iz_size = 1;
#endif

        for(int ibatch=iz_start; ibatch <= batchCount; ibatch += iz_size) {
                T const * const A_ = Aarray(ibatch);
                T const * const B_ = Barray(ibatch);
                T*        const C_ = Carray(ibatch);
                int const ldA = ldAarray(ibatch);
                int const ldB = ldBarray(ibatch);
                int const ldC = ldCarray(ibatch);

                kgemm_nt( mm,nn,kk,  alpha, A_, ldA, B_, ldB, 
                                     beta,  C_, ldC );
        };
}


#undef GLOBAL
#undef Aarray
#undef Barray
#undef Carray
#undef ldAarray
#undef ldBarray
#undef ldCarray


#endif
