#ifndef KRONMULT2_HPP
#define  KRONMULT2_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmult1.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,A2)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = A2*W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronmult2( int const n, 
                int const nvec_in,
                T   const A1_[],
                T   const A2_[],
                T   X_[],
                T   Y_[],
                T   W_[],
		int const lda_in = 0
	        )
// -----------------
// note A1 is n by n
//      X is (n^3 by nvec_in)
// -----------------
{
    int const lda = (lda_in == 0) ? n : lda_in;
    int const n2 = n*n;

    int const ldX = n2;
    int const ldW = n2;
    int const ldY = n2;

    auto X = [=] (int const i,
                  int const j) -> T& {
            return( X_[ indx2f(i,j,ldX) ] );
    };

    auto W = [=] (int const i,
                  int const j) -> T& {
            return( W_[ indx2f(i,j,ldW) ] );
    };

    auto Y = [=] (int const i,
                  int const j) -> T& {
            return( Y_[ indx2f(i,j,ldY) ] );
    };

    // --------------------------------------------------
    // perform blocking to reuse only a small part of "W"
    // to encourage better cache reuse
    // --------------------------------------------------
    int const cache_size = 32*1024;
    int const nb_cache = cache_size/(2*n*n*sizeof(T));
    int const nb = max(1, nb_cache );
    for(int istart=1; istart <= nvec_in; istart += nb) {
        int const iend = min( nvec_in, istart + nb - 1);
        int const nvec = iend - istart + 1;

        for(int i=istart; i <= iend; i++) {
            T * const Xi_ = &( X(1, i) );
            T * const Wi_ = &( W(1, 1+(i-istart)));

            int const ldXi = n;
            int const ldWi = n;
            // ----------------------------
            // Xi viewed as (n by n) array
            // Wi viewed as (n by n) array
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
            // Wi(1:n, 1:n) = Xi(1:n, 1:n) * transpose(A1(1:n,1:n))
            // --------------------------------------------------------
            int const mm = n;
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
    }; // for(i)

    int const next_nvec = nvec * n;

    // --------------------------------
    // note now X_ is used as workspace
    // --------------------------------
    {
    int const i = istart;
    T * const Xi_ = &( X(1, i) );
    T * const Wi_ = &( W(1, 1) );

    T * const Yi_ = &( Y(1, i) );

    kronmult1( n, next_nvec, 
               A2_, 
               Wi_,  Yi_,   Xi_, lda );
    };

   }; // for(istart)

}




#endif
