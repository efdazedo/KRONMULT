#ifndef KRONMULTV6_HPP
#define  KRONMULTV6_HPP 1

#include "kroncommon.hpp"

#include "kronmultv.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A6) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmultv6( int const m1, int const n1, T const A1_[], int const ld1,
		 int const m2, int const n2, T const A2_[], int const ld2,
		 int const m3, int const n3, T const A3_[], int const ld3,
		 int const m4, int const n4, T const A4_[], int const ld4,
		 int const m5, int const n5, T const A5_[], int const ld5,
		 int const m6, int const n6, T const A6_[], int const ld6,
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
//      A5 is m5 by n5
//      A6 is m6 by n6
//      X is (n1*n2*n3*n4*n5*n6) by nvec
//      Y is (m1*m2*m3*m4*m5*m6) by nvec
//      Y = kron(A1,A2,A3,A4,A5)*X
// -----------------
{
    int constexpr ndim = 6;
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
