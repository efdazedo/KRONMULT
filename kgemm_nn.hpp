#ifndef KGEMM_NN_HPP
#define KGEMM_NN_HPP 1

#include "kroncommon.hpp"


//  -----------------------
//  NotransA and NotransB case
//  C = alpha*A*B + beta *C
//  -----------------------
template<typename T>
DEVICE_FUNCTION
void kgemm_nn( int const mm, int const nn, int const kk, 
               T const alpha,
               T const * const A_,  int const ldA,
               T const * const B_,  int const ldB,
               T const beta,
               T * C_,  int const ldC)
{
#ifdef USE_GPU
        // ---------------------------
        // use matlab 1 based indexing
        // ---------------------------
        int const ix_start = threadIdx.x + 1;
        int const ix_size = blockDim.x;
        int const iy_start = threadIdx.y + 1;
        int const iy_size = blockDim.y;
#else
        int const ix_start = 1;
        int const ix_size = 1;
        int const iy_start = 1;
        int const iy_size = 1;
#endif

        assert( ix_start >= 1);
        assert( iy_start >= 1);
        assert( ix_size >= 1 );
        assert( iy_size >= 1 );

        int constexpr nb = 16;

        int constexpr total_cache = 3 * nb * nb;
        SHARED_MEMORY T cache_memory[total_cache];

        int const nb_m = MIN( nb, mm );
        int       nb_n = MIN( nb, nn );
        int const nb_k = MIN( nb, kk );

        //  ------------------------------------
        //  commonly  nn is large, but mm, kk are small
        //
        //  consider increasing nb_n for more effective
        //  use of shared cache
        //
        //  nb_m * nb_k is storage for Atmp
        //  nb_k * nb_n is storage for Btmp
        //  nb_m * nb_n is storage for Ctmp
        //  cache_memory = nb_m*nb_n + nb_k*nb_n + nb_m*nb_k
        //  ------------------------------------
        nb_n = (total_cache - nb_m*nb_k)/( nb_m + nb_k);
        // -------------------------
        // make nb_n a multple of nb
        // -------------------------
        nb_n = nb * MAX(1, nb_n/nb);

        int ifree = 0;
        int const ip_Atmp = ifree; ifree += nb_m * nb_k;
        int const ip_Ctmp = ifree; ifree += nb_m * nb_n;
        int const ip_Btmp = ifree; ifree += nb_k * nb_n;
        ifree = 0;

#define A(ia,ja)  A_[ indx2f(ia,ja,ldA) ]
#define B(ib,jb)  B_[ indx2f(ib,jb,ldB) ]
#define C(ic,jc)  C_[ indx2f(ic,jc,ldC) ]

#define Atmp(i,j)  cache_memory[ ip_Atmp + indx2f(i,j,nb_m) ]
#define Btmp(i,j)  cache_memory[ ip_Btmp + indx2f(i,j,nb_k) ]
#define Ctmp(i,j)  cache_memory[ ip_Ctmp + indx2f(i,j,nb_m) ]

    bool const Atmp_fit = (mm <= nb_m) && (kk <= nb_k);
    bool const need_load_Atmp = !Atmp_fit;
    if (Atmp_fit) {
            //  --------------------------
            //  just load A into Atmp shared memory cache once
            //  --------------------------
            for(int j=iy_start; j <= kk; j += iy_size) {
            for(int i=ix_start; i <= mm; i += ix_size) {
                Atmp(i,j) = A(i,j);
            };
            };
      };


        for(int jstart=1; jstart <= nn; jstart += nb_n) {
            int const jend = MIN(nn, jstart + nb_n-1);
            int const jsize = jend  - jstart + 1;

            for(int istart=1; istart <= mm;  istart += nb_m) {
                int const iend = MIN( mm, istart + nb_m-1);
                int const isize = iend - istart + 1;

                SYNCTHREADS;

                for(int j=iy_start; j <= jsize; j += iy_size) {
                for(int i=ix_start; i <= isize; i += ix_size) {
                  Ctmp(i,j)  = 0;
                };
                };

                SYNCTHREADS;

                for(int kstart=1; kstart <= kk; kstart += nb_k) {
                    int const kend = MIN(kk, kstart+nb_k-1);
                    int const ksize = kend - kstart + 1;

                    // ----------------------------------------------------------
                    // load Atmp(1:isize,1:ksize) <- A( istart:iend, kstart:kend)
                    // load Btmp(1:ksize,1:jsize) <- B( kstart:kend, jstart:jend)
                    // ----------------------------------------------------------
        
                    if (need_load_Atmp) {
                     for(int k=iy_start; k <= ksize; k += iy_size) {
                     for(int i=ix_start; i <= isize; i += ix_size) {
                       Atmp(i,k) = A( (istart-1) + i, (kstart-1) + k);
                       };
                       };
                    };

                      SYNCTHREADS;


                    for(int j=iy_start; j <= jsize; j += iy_size) {
                    for(int k=ix_start; k <= ksize; k += ix_size) {
                       Btmp(k,j) = B( (kstart-1) + k, (jstart-1) + j);
                       };
                       };

                    SYNCTHREADS;


                    // ---------------------------
                    // perform matrix calculations
                    // ---------------------------
                    
                    for(int j=iy_start; j <= jsize; j += iy_size) {
                    for(int i=ix_start; i <= isize; i += ix_size) {
                            for(int k=1; k <= ksize; k++) {
                                Ctmp(i,j) += Atmp(i,k) * Btmp(k,j);
                            };
                    };
                    };

                    SYNCTHREADS;

                  }; // end for kstart

                SYNCTHREADS;
                // ------------------
                // store results to C
                // ------------------
                for(int j=iy_start; j <= jsize; j += iy_size) {
                for(int i=ix_start; i <= isize; i += ix_size) {
                      int const ic = (istart-1) + i;
                      int const jc = (jstart-1) + j;
                      C(ic,jc) = alpha*Ctmp(i,j) + beta * C(ic,jc);
                };
                };

                SYNCTHREADS;
            }; // end istart
        }; // end jstart
}










#undef A
#undef B
#undef C
#undef Atmp
#undef Btmp
#undef Ctmp

#endif
