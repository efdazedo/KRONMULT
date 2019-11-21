#include "kronmult6_batched.hpp"

void kronmult6_batched(
                       int const n,
                       double const Aarray_[],
                       double Xarray_[],
                       double Yarray_[],
                       double Warray_[],
                       int const batchCount )
{
        kronmult6_batched<double>( n, 
           Aarray_, Xarray_, Yarray_, Warray_, batchCount);

}


