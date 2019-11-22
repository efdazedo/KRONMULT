#ifndef KGEMM_NT_HPP
#define KGEMM_NT_HPP 1

#include "kroncommon.hpp"


//  -----------------------
//  NotransA and TransB case
//  C = alpha*A*transpose(B) + beta *C
//  -----------------------
template<typename T>
DEVICE_FUNCTION
void kgemm_nt( int const mm, int const nn, int const kk, 
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
        int constexpr warpsize = 32;
        int const nthreads = blockDim.x; 
        assert( blockDim.y == 1);
        assert( blockDim.z == 1);
        assert( MOD(nthreads, warpsize) == 0);

        // -----------------------------------------
        // reorganize threads as nx_threads by ny_threads
        // -----------------------------------------
        int const nx_threads = warpsize;
        int const ny_threads = nthreads/nx_threads;

        int const ix_start = MOD( threadIdx.x, nx_threads ) + 1;
        int const iy_start = (threadIdx.x/nx_threads) + 1;

        int const ix_size = nx_threads;
        int const iy_size = ny_threads;
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
        int constexpr total_cache = 3*nb*nb;
        SHARED_MEMORY T cache_memory[ total_cache ];

        int nb_m = MIN(mm, nb);
        int nb_n = MIN(nn, nb);
        int nb_k = MIN(kk, nb);

        //  ------------------------------------
        //  commonly  mm is large, but kk, nn are small
        //
        //  consider increasing nb_m for more effective
        //  use of shared cache
        //
        //  nb_m * nb_k is storage for Atmp
        //  nb_n * nb_k is storage for Btmp
        //  nb_m * nb_n is storage for Ctmp
        //  cache_memory = nb_m*nb_n + nb_n*nb_k + nb_m*nb_k
        //  ------------------------------------
        nb_m = (total_cache - nb_n*nb_k)/( nb_n + nb_k);
        // -------------------------
        // make nb_m a multiple of nb
        // -------------------------
        nb_m = nb * MAX(1, nb_m/nb);

        int ifree = 0;
        int const ip_Btmp = ifree; ifree += nb_n * nb_k;
        int const ip_Ctmp = ifree; ifree += nb_m * nb_n;
        int const ip_Atmp = ifree; ifree += nb_m * nb_k;

#define A(ia,ja)  A_[ indx2f(ia,ja,ldA) ]
#define B(ib,jb)  B_[ indx2f(ib,jb,ldB) ]
#define C(ic,jc)  C_[ indx2f(ic,jc,ldC) ]

#define Atmp(i,j)  cache_memory[ ip_Atmp + indx2f(i,j,nb_m) ]
#define Ctmp(i,j)  cache_memory[ ip_Ctmp + indx2f(i,j,nb_m) ]
#define Btmp(i,j)  cache_memory[ ip_Btmp + indx2f(i,j,nb_n) ]

        bool const is_Btmp_fit = (kk <= nb_k) && (nn <= nb_n);
        bool const need_load_Btmp = !is_Btmp_fit;
        if (is_Btmp_fit) {
                // ------------------------------------------
                // load B only once into Btmp in shared cache 
                // ------------------------------------------
                for(int k=iy_start; k <= kk; k += iy_size) {
                for(int j=ix_start; j <= nn; j += ix_size) {
                         Btmp(j,k) = B(j, k);
                       };
                       };
        };

        for(int jstart=1; jstart <= nn; jstart += nb) {
            int const jend = MIN(nn, jstart + nb-1);
            int const jsize = jend  - jstart + 1;

            for(int istart=1; istart <= mm;  istart += nb) {
                int const iend = MIN( mm, istart + nb-1);
                int const isize = iend - istart + 1;

                SYNCTHREADS;

                for(int j=iy_start; j <= jsize; j += iy_size) {
                for(int i=ix_start; i <= isize; i += ix_size) {
                  Ctmp(i,j)  = 0;
                };
                };

                SYNCTHREADS;

                for(int kstart=1; kstart <= kk; kstart += nb) {
                    int const kend = MIN(kk, kstart+nb-1);
                    int const ksize = kend - kstart + 1;

                    // ----------------------------------------------------------
                    // load Atmp(1:isize,1:ksize) <- A( istart:iend, kstart:kend)
                    // load Btmp(1:jsize,1:ksize) <- B( jstart:jend, kstart:kend)
                    // ----------------------------------------------------------
        
                    for(int k=iy_start; k <= ksize; k += iy_size) {
                    for(int i=ix_start; i <= isize; i += ix_size) {
                       Atmp(i,k) = A( (istart-1) + i, (kstart-1) + k);
                       };
                       };

                    SYNCTHREADS;


                    if (need_load_Btmp) {
                      for(int k=iy_start; k <= ksize; k += iy_size) {
                      for(int j=ix_start; j <= jsize; j += ix_size) {
                         Btmp(j,k) = B( (jstart-1) + j, (kstart-1) + k);
                       };
                       };
                    };

                    SYNCTHREADS;


                    // ---------------------------
                    // perform matrix calculations
                    // ---------------------------
                    
                    for(int j=iy_start; j <= jsize; j += iy_size) {
                    for(int i=ix_start; i <= isize; i += ix_size) {
                            for(int k=1; k <= ksize; k++) {
                                Ctmp(i,j) += Atmp(i,k) * Btmp(j,k);
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
