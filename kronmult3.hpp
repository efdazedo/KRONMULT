#ifndef KRONMULT3_HPP
#define  KRONMULT3_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A3)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A3) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmult3( int const n, 
                int const nvec,
                T   const A1_[],
                T   const A2_[],
                T   const A3_[],
                T   X_[],
                T   Y_[],
                T   W_[],
	        int const lda_in = 0 )
// -----------------
// note A1 is n by n
//      X is (n^3 by nvec)
// -----------------
{
    int constexpr ndim = 3;
    kronmultx<T,ndim>(
                      n, nvec,
                      A1_,A2_,A3_,nullptr,nullptr,nullptr,
                      X_,Y_,W_,lda_in );
}

#endif
