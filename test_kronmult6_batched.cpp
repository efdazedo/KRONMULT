#include <iostream>
#include <cassert>

#include "kroncommon.hpp"
#include "kronmult6_batched.hpp"


#ifdef USE_GPU
#include <cuda_runtime.h>
#else
#include <stdlib.h>
#endif

static inline
void *myalloc( size_t nbytes ) {
              void *devPtr = nullptr;
#ifdef USE_GPU
        // -------------------------
        // use unified shared memory
        // for simplicity
        // -------------------------
              cudaError_t istat = cudaMallocManaged( &devPtr, nbytes );
              assert( istat == cudaSuccess );
#else
              devPtr = malloc( nbytes );
#endif
              assert( devPtr != nullptr );
              return(devPtr);
}

static inline
void myfree( void * devPtr ) {
#ifdef USE_GPU
                cudaError_t istat = cudaFree( devPtr);
                assert( istat == cudaSuccess );
#else
                free( devPtr );
#endif
}
     

template<typename T>
T test_kronmult6_batched( int const n, int const batchCount )
{
        int const n2 = n*n;
        int const n3 = n*n2;
        int const n4 = n*n3;
        int const n5 = n*n4;
        int const n6 = n*n5;




        // -------------------------
        // Aarray is (n,n,6,batchCount)
        // Xarray is (n6 by batchCount)
        // Yarray is (n6 by batchCount)
        // Warray is (n6 by batchCount)
        // ----------------------------



        T *Aarray_ = (T *) myalloc( sizeof(T)*n*n*6*batchCount);
        T *Xarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );
        T *Yarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );
        T *Warray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );


#define Aarray(i,j,k,ibatch) Aarray_[ indx4f(i,j,k,ibatch, n,n,6) ]
#define Xarray(i,ibatch) Xarray_[ indx2f(i,ibatch,n6) ]
#define Yarray(i,ibatch) Yarray_[ indx2f(i,ibatch,n6) ]


        //  ---------------------
        //  initialize the arrays
        //  ---------------------
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
        for(int i=1; i <= n6; i++) {
              T const r1 = (i + (ibatch-1)*n6 );
              T const r2 = n6*batchCount;
              Xarray(i,ibatch) = r1/r2;
              };
              };
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
            for(int k=1; k <= 6; k++) {
            for(int j=1; j <= n; j++) {
            for(int i=1; i <= n; i++) {
                T const r1 = (i + (j-1)*n + (k-1)*n*n + ibatch);
                T const r2 = n*n*n;
                Aarray(i,j,k,  ibatch) = r1/r2;
            };
            };
            };
        };




        {
        dim3 grid(batchCount,1,1);
        dim3 block(16,16,1);

        kronmult6_batched<T><<< grid, block >>>( n,
                           Aarray_,
                           Xarray_,
                           Yarray_,
                           Warray_,
                           batchCount );
        }



        // -------------
        // check results
        // -------------
        T max_abserr = 0;
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                T const * const A1_ = &(Aarray(1,1,1,ibatch));
                T const * const A2_ = &(Aarray(1,1,2,ibatch));
                T const * const A3_ = &(Aarray(1,1,3,ibatch));
                T const * const A4_ = &(Aarray(1,1,4,ibatch));
                T const * const A5_ = &(Aarray(1,1,5,ibatch));
                T const * const A6_ = &(Aarray(1,1,6,ibatch));
                T const * const X_ = &(Xarray(1,ibatch));
                T       * const Y_ = &(Yarray(1,ibatch));

#define X(ic) X_[ (ic)-1 ]
#define Y(ic) Y_[ (ic)-1 ]

#define A1(i,j) A1_[ indx2f(i,j,n) ]
#define A2(i,j) A2_[ indx2f(i,j,n) ]
#define A3(i,j) A3_[ indx2f(i,j,n) ]
#define A4(i,j) A4_[ indx2f(i,j,n) ]
#define A5(i,j) A5_[ indx2f(i,j,n) ]
#define A6(i,j) A6_[ indx2f(i,j,n) ]



                for(int i1=1; i1 <= n; i1++) {
                for(int i2=1; i2 <= n; i2++) {
                for(int i3=1; i3 <= n; i3++) {
                for(int i4=1; i4 <= n; i4++) {
                for(int i5=1; i5 <= n; i5++) {
                for(int i6=1; i6 <= n; i6++) {

                   int const ic = indx6f( i6,i5,i4,i3,i2,i1,n,n,n,n,n);
                   T Y_ic = 0;

                   for(int j1=1; j1 <= n; j1++) {
                   for(int j2=1; j2 <= n; j2++) {
                   for(int j3=1; j3 <= n; j3++) {
                   for(int j4=1; j4 <= n; j4++) {
                   for(int j5=1; j5 <= n; j5++) {
                   for(int j6=1; j6 <= n; j6++) {

                      // -------------------------------
                      // note last index i6 goes fastest
                      // -------------------------------
                      int const jc = indx6f( j6,j5,j4,j3,j2,j1,n,n,n,n,n);

                      T const C_ic_jc = A1(i1,j1)*A2(i2,j2)*A3(i3,j3)*A4(i4,j4)*A5(i5,j5)*A6(i6,j6);

                      T const X_jc = X(jc);

                      Y_ic += C_ic_jc * X_jc;
                   };
                   };
                   };
                   };
                   };
                   };

                   T const abs_err = ABS( Y_ic - Y(ic) );
                   max_abserr = MAX( max_abserr, abs_err );

                };
                };
                };
                };
                };
                };
       }; // end for ibatch



        // -------
        // cleanup
        // -------

        myfree( Aarray_ ); Aarray_ = nullptr;
        myfree( Xarray_ ); Xarray_ = nullptr;
        myfree( Yarray_ ); Yarray_ = nullptr;
        myfree( Warray_ ); Warray_ = nullptr;

        return(max_abserr);
#undef X
#undef Y

#undef Aarray
#undef Xarray
#undef Yarray
}


                      
int main() {

        int const inc = 5;
        int const batchCount_max = 10;

        int nerrors = 0;
        for(int batchCount=1; batchCount <= batchCount_max; batchCount += inc) {
        for(int n=1; n <= 8; n++) {
                double const max_abserr =  test_kronmult6_batched<double>(n, batchCount );
                double const tol = 1.0/(1000.0 * 1000.0);
                bool const isok = (max_abserr <= tol);
                if (!isok) {
                        nerrors += 1;
                };
        };
        };


        if (nerrors == 0) {
                std::cout << "ALL PASSED" << "\n";
        }
        else {
                std::cout << "There are " << nerrors << " errors" << "\n";
        };

  return(0);
}


                     


