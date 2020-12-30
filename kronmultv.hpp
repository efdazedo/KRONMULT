#ifndef KRONMULTV_HPP
#define  KRONMULTV_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kgemm_nn.hpp"





//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A6) * W
//  -------------------------------------------
template<typename T, int ndim=1>
DEVICE_FUNCTION
void kronmultv( 
                
                int const m1, int const n1, T   const A1_[], int const ld1,
                int const m2, int const n2, T   const A2_[], int const ld2,
                int const m3, int const n3, T   const A3_[], int const ld3,
                int const m4, int const n4, T   const A4_[], int const ld4,
                int const m5, int const n5, T   const A5_[], int const ld5,
                int const m6, int const n6, T   const A6_[], int const ld6,
		int const nvec,
                T   X_[],
                T   Y_[],
                T   W_[]
	        )
{
  // -----------------
  // note A1 is m1 by n1
  //      X is (n1*...*n6) by nvec or
  //      or (n1*...*n5) by (n6*nvec)
  //      W is (n2*...n6) by (n1*nvec)
  //      Y is (m1*...*m6) by nvec
  //      or (m2*...*m6) * (m1*nvec)
  // -----------------
  
   assert( (1 <= m1) && (1 <= n1) && (m1 <= ld1) );
   assert( (1 <= m2) && (1 <= n2) && (m2 <= ld2) );
   assert( (1 <= m3) && (1 <= n3) && (m3 <= ld3) );
   assert( (1 <= m4) && (1 <= n4) && (m4 <= ld4) );
   assert( (1 <= m5) && (1 <= n5) && (m5 <= ld5) );
   assert( (1 <= m6) && (1 <= n6) && (m6 <= ld6) );
  

  
      int const n26 = (n2*n3)*(n4*n5)*n6;
      int const ldX = (n1*n26);
      int const ldW = (   n26) * m1;
  
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
              int const ldXi = n26;
              int const ldWi = ldXi;
              // ----------------------------
              // Xi viewed as (n2*...*n6) by (n1) array
              // Wi viewed as (n2*...*n6) by (m1) array
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
              // Wi(1:(n2*...*n6), 1:m1) = Xi(1:(n2*...*n6), 1:n1) * transpose(A1(1:m1,1:n1))
              // --------------------------------------------------------
              int const mm = n26;
              int const nn = m1;
              int const kk = n1;
              T const alpha = 1;
              T const beta = 0;
  
              T const * const  Ap = &(Xi(1,1));
              T const * const  Bp = A1_;
              T       * const  Cp = &(Wi(1,1));

	      assert( Ap != nullptr );
	      assert( Bp != nullptr );
	      assert( Cp != nullptr );
  
              int const ldAp = ldXi;
              int const ldBp = ld1;
              int const ldCp = ldWi;

  
              kgemm_nt( mm,nn,kk, 
                        alpha, Ap, ldAp,
                               Bp, ldBp,
                        beta,  Cp, ldCp );
      };
  
      int const next_nvec = nvec * m1;
  
      // --------------------------------
      // note now X_ is used as workspace
      // --------------------------------
      {
	kronmultv<T,ndim-1>( 
			n2,m2,A2_,ld2,
			n3,m3,A3_,ld3,
			n4,m4,A4_,ld4,
			n5,m5,A5_,ld5,
			n6,m6,A6_,ld6,
			1, 1, nullptr,1,  // unused
			next_nvec,
			W_, Y_, X_ );
      }
  
}



template<>
DEVICE_FUNCTION
void kronmultv( 
               int const m1, int const n1, double const A1_[], int const ld1,
               int const m2, int const n2, double const A2_[], int const ld2,
               int const m3, int const n3, double const A3_[], int const ld3,
               int const m4, int const n4, double const A4_[], int const ld4,
               int const m5, int const n5, double const A5_[], int const ld5,
               int const m6, int const n6, double const A6_[], int const ld6,
	       int const nvec,
               double   X_[],
               double   Y_[],
               double   W_[]
	      )
{

   assert( (1 <= m1) && (1 <= n1) && (m1 <= ld1) );

    // ---------------------------------------------------
    // Y(1:m1, 1:nvec ) += A1(1:m1, 1:n1) * X(1:n1, 1:nvec) 
    // ---------------------------------------------------
    int const mm = m1;
    int const nn = nvec;
    int const kk = n1;
    double const alpha = 1;
    double const beta = 1;
    int const ldAp = ld1;
    int const ldBp = n1;
    int const ldCp = ld1;

    double const * const Ap = &(A1_[0]);
    double const * const Bp = &(X_[0]);
    double       * const Cp = &(Y_[0]);

    assert( Ap != nullptr );
    assert( Bp != nullptr );
    assert( Cp != nullptr );

    kgemm_nn( mm,nn,kk,
              alpha,  Ap, ldAp,
                      Bp, ldBp,
              beta,   Cp, ldCp );
}



template<>
DEVICE_FUNCTION
void kronmultv( 
               int const m1, int const n1, float const A1_[], int const ld1,
               int const m2, int const n2, float const A2_[], int const ld2,
               int const m3, int const n3, float const A3_[], int const ld3,
               int const m4, int const n4, float const A4_[], int const ld4,
               int const m5, int const n5, float const A5_[], int const ld5,
               int const m6, int const n6, float const A6_[], int const ld6,
	       int const nvec,
               float   X_[],
               float   Y_[],
               float   W_[]
	      )
{

    // ---------------------------------------------------
    // Y(1:m1, 1:nvec ) += A1(1:m1, 1:n1) * X(1:n1, 1:nvec) 
    // ---------------------------------------------------
    int const mm = m1;
    int const nn = nvec;
    int const kk = n1;
    float const alpha = 1;
    float const beta = 1;
    int const ldAp = ld1;
    int const ldBp = n1;
    int const ldCp = ld1;

    float const * const Ap = &(A1_[0]);
    float const * const Bp = &(X_[0]);
    float       * const Cp = &(Y_[0]);

    kgemm_nn( mm,nn,kk,
              alpha,  Ap, ldAp,
                      Bp, ldBp,
              beta,   Cp, ldCp );
}




#endif
