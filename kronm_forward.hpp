
#ifndef KRONM_FORWARD_HPP
#define  KRONM_FORWARD_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"





//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X 
//
//  using forward variant in Algorithm 993
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronm_forward( 
        int const ndim,
        int const m_array[],
        int const n_array[],
        T const * const A_array[],
        int const nvec,
        T* X_,
        T* Y_,
        T* W_ 
        )
                
{
    auto transpose_copy = [] (int const m_dest, int const n_dest,
                              T const * const Asrc_,  int const ldA,
                              T       * const Bdest_, int const ldB ) {
        // for(j=1; j <= n_dest; j++) {
        // for(i=1; i <= m_dest; i++) {
        //   Bdest(i,j) = Asrc(j,i);
        //   };
        //   };
            int const ij_end = (m_dest * n_dest);

#ifdef USE_GPU
	    int const ij_start = threadIdx.x + 1;
	    int const ij_inc = nthreads;
#else
            int const ij_start = 1;
            int const ij_inc = 1;
#endif

            SYNCTHREADS;

            for(int ij=ij_start; ij <= ij_end; ij += ij_inc) {
                // ij = i + (j-1)*mm
                int const i = (ij-1) % m_dest + 1;
                int const j = (ij - i)/m_dest + 1;

                Bdest_[ indx2f(i,j,ldB) ] = Asrc_[ indx2f(j,i,ldA) ];
            };

            SYNCTHREADS;
    };

    auto prod = [] (int const ndim, int const n_array[]) -> int {
        int iprod = 1;
        for(int i=0; i < ndim; i++) {
            iprod *= n_array[i];
        };
        return(iprod);
    };


    T *Xin = nullptr;
    T *Yout = nullptr;
    int const Xsize = prod(ndim,n_array );
    int Xin_size = nvec * Xsize;

    bool const need_transpose = (nvec > 1);
    if (need_transpose) {
        // ------------------------------
        // need to perform transpose copy
        // from (Xsize by nvec) to (nvec by Xsize);
        // ------------------------------
        int const m_dest = nvec;
        int const n_dest = prod( ndim, n_array );

        T const * const Asrc = X_;
        T       * const Bdest = W_;
        int const ldAsrc = n_dest;
        int const ldBdest = m_dest;

        transpose_copy( m_dest,n_dest, Asrc, ldAsrc, Bdest, ldBdest );
        Xin = Bdest;
        Yout = X_;
    }
    else {
       // ----------------------------
       // no need to perform transpose
       // ----------------------------
       Xin = X_;
       Yout = W_;
    };


    for( int i=0; i < ndim; i++) {


        bool const is_final = (i == (ndim-1));

        T const alpha = 1;

        
        T const * const Ap = A_array[i];
        T const * const Bp = Xin;

        T * Cp = (is_final) ? Y_ : Yout;
        T const beta = (is_final) ? 1 : 0;

          // -----------------
          // Yout = Amat * Xin'
          // Amat is m by k, Xin is n by k
          // -----------------

          int const mm = m_array[i];
          int const kk = n_array[i];

          assert( kk > 0 );
          int const nn = Xin_size / kk;
          assert( nn * kk == Xin_size );

          int const ld1 = mm;
          int const ld2 = nn;
          int const ld3 = mm;

          kgemm_nt(mm,nn,kk,
                  alpha, 
                  Ap, ld1, Bp, ld2,
                  beta, 
                  Cp, ld3 );
          // -----------------
          // swap Xin and Yout
          // -----------------
          {
          T * temp = Xin;
          Xin = Yout;
          Yout = temp;
          };
          Xin_size = mm * nn;
        }; // for i
}
#endif
