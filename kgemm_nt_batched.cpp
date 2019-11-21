#include "kgemm_nt_batched.hpp"


#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

#define GLOBAL  __global__ 
#else
#define GLOBAL
#endif

void kgemm_nt_batched( int const mm, int const nn, int const kk, 
                       double const alpha, 
                       double* const Aarray_[], 
                       int const ldAarray_[], 
                       double* const Barray_[], 
                       int const ldBarray_[], 
                       double const beta,  
                       double* const Carray_[], 
                       int const ldCarray_[], 
                       int const batchCount)
{
#ifdef USE_GPU
        dim3 grid(batchCount,1,1);
        dim3 block(16,16,1);

        kgemm_nt_batched<double><<< grid, block>>>( mm,nn,kk,
                          alpha,
                          Aarray_, ldAarray_,
                          Barray_, ldBarray_,
                          beta,
                          Carray_, ldCarray_,
                          batchCount );
#else
        kgemm_nt_batched<double>( mm,nn,kk,
                          alpha,
                          Aarray_, ldAarray_,
                          Barray_, ldBarray_,
                          beta,
                          Carray_, ldCarray_,
                          batchCount );
#endif
}
