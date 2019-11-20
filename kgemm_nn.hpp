#ifndef KGEMM_NN_HPP
#define KGEMM_NN_HPP 1

#include <cassert>


#ifndef MAX
#define MAX(x,y)  ( ((x) > (y)) ? (x) : (y) )
#endif

#ifndef MIN
#define MIN(x,y)  (  ((x) < (y)) ? (x) : (y) )
#endif

#ifndef indx2f
#define indx2f(i,j,ld) (((i)-1) + ((j)-1)*(ld))
#endif

#ifdef USE_GPU
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE __device__
#else
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE 
#endif

//  -----------------------
//  NotransA and NotransB case
//  C = alpha*A*B + beta *C
//  -----------------------
template<typename T>
DEVICE
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

        SHARED_MEMORY T Atmp_[nb*nb];
        SHARED_MEMORY T Btmp_[nb*nb];
        SHARED_MEMORY T Ctmp_[nb*nb];

#define A(ia,ja)  A_[ indx2f(ia,ja,ldA) ]
#define B(ib,jb)  B_[ indx2f(ib,jb,ldC) ]
#define C(ic,jc)  C_[ indx2f(ic,jc,ldB) ]

#define Atmp(i,j)  Atmp_[ indx2f(i,j,nb) ]
#define Btmp(i,j)  Btmp_[ indx2f(i,j,nb) ]
#define Ctmp(i,j)  Ctmp_[ indx2f(i,j,nb) ]


        for(int jstart=1; jstart <= nn; jstart += nb) {
            int const jend = MIN(nn, jstart + nb-1);
            int const jsize = jend  - jstart + 1;

            for(int istart=1; istart <= mm;  istart += nb) {
                int const iend = MIN( mm, istart + nb-1);
                int const isize = iend - istart + 1;

                SYNCTHREADS;

                for(int j=iy_start; j <= jsize; j += iy_size) {
                for(int i=ix_start; i <= jsize; i += ix_size) {
                  Ctmp(i,j)  = 0;
                };
                };

                SYNCTHREADS;

                for(int kstart=1; kstart <= kk; kstart += nb) {
                    int const kend = MIN(kk, kstart+nb-1);
                    int const ksize = kend - kstart + 1;

                    // ----------------------------------------------------------
                    // load Atmp(1:isize,1:ksize) <- A( istart:iend, kstart:kend)
                    // load Btmp(1:ksize,1:jsize) <- B( kstart:kend, jstart:jend)
                    // ----------------------------------------------------------
        
                    for(int k=iy_start; k <= jsize; k += iy_size) {
                    for(int i=ix_start; i <= isize; i += ix_size) {
                       Atmp(i,k) = A( (istart-1) + i, (kstart-1) + k);
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










#undef MAX
#undef MIN
#undef indx2f
#undef SYNCTHREADS 
#undef SHARED_MEMORY 
#undef DEVICE 
#undef A
#undef B
#undef C
#undef Atmp
#undef Btmp
#undef Ctmp

#endif
