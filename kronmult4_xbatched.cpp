#include "kronmult4_xbatched.hpp"

void kronmult4_xbatched(
                       int const n,
                       double const Aarray_[],
                       double* Xarray_[],
                       double* Yarray_[],
                       double* Warray_[],
                       int const batchCount )
{
#ifdef USE_GPU
        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        kronmult4_xbatched<double><<< batchCount, nthreads >>>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#else
        kronmult4_xbatched<double>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);
#endif

}


