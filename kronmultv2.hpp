#ifndef KRONMULTV2_HPP
#define  KRONMULTV2_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultv.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A2)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A2) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmultv2( int const m1, int const n1, T const A1_[], int const ld1,
		 int const m2, int const n2, T const A2_[], int const ld2,
                int const nvec,
                T   X_[],
                T   Y_[],
                T   W_[]
	        )
// -----------------
// note A1 is m1 by n1
//      A2 is m2 by n2
//      X is (n1*n2) by nvec
//      Y is (m1*m2) by nvec
//      Y = kron(A1,A2)*X
// -----------------
{
    int constexpr ndim = 2;
    int const m3 = 1; int n3 = 1; T const * const A3_ = nullptr; int const ld3 = 1;
    int const m4 = 1; int n4 = 1; T const * const A4_ = nullptr; int const ld4 = 1;
    int const m5 = 1; int n5 = 1; T const * const A5_ = nullptr; int const ld5 = 1;
    int const m6 = 1; int n6 = 1; T const * const A6_ = nullptr; int const ld6 = 1;
    kronmultv<T,ndim>(
		    m1,n1,A1_,ld1,
		    m2,n2,A2_,ld2,
		    m3,n3,A3_,ld3,
		    m4,n4,A4_,ld4,
		    m5,n5,A5_,ld5,
		    m6,n6,A6_,ld6,
		    nvec,
                    X_,Y_,W_);
}

#endif
