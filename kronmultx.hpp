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
#if (0) 
	auto pow = [=](int const x,
		       int const d) -> int {
		// compute x^d
                assert( d >= 0);
		int result = 1;
		for(int i=0; i < d; i++) {
			result *= x;
		};
		return(result);
	};
      int const n_to_ndim1 = pow(n,ndim-1);
      int const n_to_ndim = n * n_to_ndim1;
  
  
      int const ldX = n_to_ndim;
      int const ldW = n_to_ndim;
  
      auto X = [=] (int const i,
                    int const j) -> T& {
              return( X_[ indx2f(i,j,ldX) ] );
      };
  
      auto W = [=] (int const i,
                    int const j) -> T& {
              return( W_[ indx2f(i,j,ldW) ] );
      };
  
  
  
  
      for(int i=1; i <= nvec; i++) {
              T *Xi_ = &( X(1, i) );
              T *Wi_ = &( W(1, i) );
              int const ldXi = n_to_ndim1;
              int const ldWi = n_to_ndim1;
              // ----------------------------
              // Xi viewed as (n^(ndim-1) by n) array
              // Wi viewed as (n^(ndim-1) by n) array
              // ----------------------------
  
              auto Xi = [=] (int const i,
                             int const j) -> T& {
                      return(  Xi_[ indx2f(i,j,ldXi) ] );
              };
  
              auto Wi = [=] (int const i,
                             int const j) -> T& {
                      return(  Wi_[ indx2f(i,j,ldWi) ] );
              };
  
              // --------------------------------------------------------
              // Wi(1:n^(ndim-1), 1:n) = Xi(1:n^(ndim-1), 1:n) * transpose(A1(1:n,1:n))
              // --------------------------------------------------------
              int const mm = n_to_ndim1;
              int const nn = n;
              int const kk = n;
              T const alpha = 1;
              T const beta = 0;
  
              T const * const  Ap = &(Xi(1,1));
              T const * const  Bp = A1_;
              T       * const  Cp = &(Wi(1,1));
  
              int const ld1 = ldXi;
              int const ld2 = lda;
              int const ld3 = ldWi;
  
              kgemm_nt( mm,nn,kk, 
                        alpha, Ap, ld1,
                               Bp, ld2,
                        beta,  Cp, ld3 );
      };
  
      int const next_nvec = nvec * n;
  
      // --------------------------------
      // note now X_ is used as workspace
      // --------------------------------
      {
        kronmultx<T,ndim-1>( n, next_nvec, 
                 A2_, A3_, A4_, A5_, A6_, A1_,
                 W_,  Y_,   X_, lda );
      }
#endif

}

#if (0)


template<>
DEVICE_FUNCTION
void kronmultx( int const n, 
                int const nvec,
                double   const A1_[],
                double   const A2_[],
                double   const A3_[],
                double   const A4_[],
                double   const A5_[],
                double   const A6_[],
                double   X_[],
                double   Y_[],
                double   W_[],
	        int const lda_in )
{
  kronmult1<double>( n, nvec,
          A1_, X_, Y_, W_, lda_in );
}




template<>
DEVICE_FUNCTION
void kronmultx( int const n, 
                int const nvec,
                float   const A1_[],
                float   const A2_[],
                float   const A3_[],
                float   const A4_[],
                float   const A5_[],
                float   const A6_[],
                float   X_[],
                float   Y_[],
                float   W_[],
	        int const lda_in )
{
  kronmult1<float>( n, nvec,
          A1_, X_, Y_, W_, lda_in );
}
#endif



#endif
