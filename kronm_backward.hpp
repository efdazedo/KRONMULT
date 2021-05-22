
#ifndef KRONM_BACKWARD_HPP
#define  KRONM_BACKWARD_HPP 1

#include "kroncommon.hpp"

#include "kgemm_tt.hpp"





//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X 
//
//  using backward variant in Algorithm 993
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronm_backward( 
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
    auto transpose_add = [] (int const m_dest, int const n_dest,
                              T const * const Asrc_,  int const ldA,
                              T       * const Bdest_, int const ldB ) {
        // for(j=1; j <= n_dest; j++) {
        // for(i=1; i <= m_dest; i++) {
        //   Bdest(i,j) += Asrc(j,i);
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

                atomicAdd( &(Bdest_[ indx2f(i,j,ldB) ]),
                             Asrc_[  indx2f(j,i,ldA) ] );
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
    int const Ysize = prod(ndim,m_array );
    int Xin_size = nvec * Xsize;


    bool const need_transpose = (nvec > 1);

    for( int i = (ndim-1); i >= 0; --i) {


        bool const is_final = (i == 0);

        T const alpha = 1;

        
          // -----------------
          // Yout = Xin' * Amat'
          // Amat is n by k, Xin is k by m
          // -----------------
        T const * const Ap = Xin;
        T const * const Bp = A_array[i];

        T * Cp = (is_final && (!need_transpose) ) ? Y_ : Yout;
        T const beta = (is_final && (!need_transpose) ) ? 1 : 0;


          int const nn = m_array[i];
          int const kk = n_array[i];

          assert( kk > 0 );
          int const mm = Xin_size / kk;
          assert( mm * kk == Xin_size );

          int const ld1 = kk;
          int const ld2 = nn;
          int const ld3 = mm;

          kgemm_tt(mm,nn,kk,
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


    if (need_transpose) {
        // Y is Ysize by nvec
        // Yout is nvec by Ysize
        // perform
        // Y(1:Ysize,1:nvec) += transpose( Yout(1:nvec,1:Ysize) )
        int const m_dest = Ysize; 
        int const n_dest = nvec;

        T const * const Asrc = Yout;
        T       * const Bdest = Y_;
        int const ldBdest = Ysize;
        int const ldAsrc = nvec;

        transpose_add( m_dest, n_dest, Asrc, ldAsrc, Bdest, ldBdest );
        };

}
#endif
