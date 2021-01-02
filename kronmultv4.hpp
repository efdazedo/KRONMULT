#ifndef KRONMULTV4_HPP
#define  KRONMULTV4_HPP 1

#include "kroncommon.hpp"

#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A4)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A4) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmultv4( int const m1, int const n1, T const A1_[], int const ld1,
		 int const m2, int const n2, T const A2_[], int const ld2,
		 int const m3, int const n3, T const A3_[], int const ld3,
		 int const m4, int const n4, T const A4_[], int const ld4,
                int const nvec,
                T   X_[],
                T   Y_[],
                T   W_[]
	        )
// -----------------
// note A1 is m1 by n1
//      A2 is m2 by n2
//      A3 is m3 by n3
//      A4 is m4 by n4
//      X is (n1*n2*n3*n4) by nvec
//      Y is (m1*m2*m3*m4) by nvec
//      Y = kron(A1,A2,A3,A4)*X
// -----------------
{
    int constexpr ndim = 4;
    int const m5 = 1; int n5 = 1; T const * const A5_ = nullptr; int const ld5 = 1;
    int const m6 = 1; int n6 = 1; T const * const A6_ = nullptr; int const ld6 = 1;
    kronmultx<T,ndim>(
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
