#ifndef KRONMULTX_HPP
#define  KRONMULTX_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmult1.hpp"

#include "kronmultv.hpp"




//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A6) * W
//  -------------------------------------------
template<typename T, int ndim=1>
DEVICE_FUNCTION
void kronmultx( int const n, 
                int const nvec,
                T   const A1_[],
                T   const A2_[],
                T   const A3_[],
                T   const A4_[],
                T   const A5_[],
                T   const A6_[],
                T   X_[],
                T   Y_[],
                T   W_[],
	        int const lda_in = 0 )
{
  // -----------------
  // note A1 is n by n
  //      X is (n^6 by nvec)
  // -----------------
        int const lda = (lda_in == 0) ? n : lda_in;
	int const m1 = n; int const n1 = n; int const  ld1 = lda;
	int const m2 = n; int const n2 = n; int const  ld2 = lda;
	int const m3 = n; int const n3 = n; int const  ld3 = lda;
	int const m4 = n; int const n4 = n; int const  ld4 = lda;
	int const m5 = n; int const n5 = n; int const  ld5 = lda;
	int const m6 = n; int const n6 = n; int const  ld6 = lda;

	kronmultv<T,ndim>(
			m1,n1,A1_,ld1,
			m2,n2,A2_,ld2,
			m3,n3,A3_,ld3,
			m4,n4,A4_,ld4,
			m5,n5,A5_,ld5,
			m6,n6,A6_,ld6,
			nvec,
			X_, Y_, W_ );

}




#endif
