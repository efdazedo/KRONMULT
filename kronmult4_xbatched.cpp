#include "hip/hip_runtime.h"
#include "kronmult4_xbatched.hpp"

void kronmult4_xbatched(
                       int const n,
                       double const * const Aarray_[],
		       int const lda,
                       double* Xarray_[],
                       double* Yarray_[],
                       double* Warray_[],
                       int const batchCount )
{
#ifdef USE_GPU
        int constexpr warpsize = WARPSIZE;
        int constexpr nwarps = 2;
        int constexpr nthreads = nwarps * warpsize;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult4_xbatched<double>), dim3(batchCount), dim3(nthreads ), 0, 0,  n, 
           Aarray_, lda, 
	   Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult4_xbatched<double>( n, 
           Aarray_, lda,
	   Xarray_, Yarray_, Warray_, batchCount);
#endif

}


