#include "hip/hip_runtime.h"
#include "kroncommon.hpp"
#include "kgemm_nn_batched.hpp"




void kgemm_nn_batched( int const mm, int const nn, int const kk, 
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
        int constexpr warpsize = WARPSIZE;
        int constexpr nwarps = 1;
        int constexpr nthreads = nwarps * warpsize;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(kgemm_nn_batched<double>), dim3(batchCount), dim3(nthreads), 0, 0,  mm,nn,kk,
                          alpha,
                          Aarray_, ldAarray_,
                          Barray_, ldBarray_,
                          beta,
                          Carray_, ldCarray_,
                          batchCount );
#else
        kgemm_nn_batched<double>( mm,nn,kk,
                          alpha,
                          Aarray_, ldAarray_,
                          Barray_, ldBarray_,
                          beta,
                          Carray_, ldCarray_,
                          batchCount );
#endif
}
