#include <iostream>
#include <cassert>
#include <chrono>
#include <unistd.h>

#include "kroncommon.hpp"
#include "kronmult6_batched.hpp"


#ifdef USE_GPU
#include <cuda_runtime.h>
#else
#include <stdlib.h>
#include <string.h>
#endif


static inline
void host2gpu( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        cudaError_t istat = cudaMemcpy( dest, 
                                        src, 
                                        nbytes,  
                                        cudaMemcpyHostToDevice );
        assert( istat == cudaSuccess );
#else
        memcpy( dest, src, nbytes );
#endif
}

static inline
void gpu2host( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        cudaError_t istat = cudaMemcpy( dest,
                                        src,
                                        nbytes,
                                        cudaMemcpyDeviceToHost);
        assert( istat == cudaSuccess );
#else
        memcpy( dest, src, nbytes );
#endif

}

static inline
void *myalloc( size_t nbytes ) {
              void *devPtr = nullptr;
#ifdef USE_GPU
              cudaError_t istat = cudaMalloc( &devPtr, nbytes );
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
T test_kronmult6_batched( int const n, int const batchCount, 
                          int const idebug = 0, 
                          bool const do_check  = true )
        
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
        // Zarray is (n6 by batchCount)
        // Warray is (n6 by batchCount)
        // ----------------------------

        T *Aarray_ = (T *) malloc( sizeof(T)*n*n*6*batchCount);
        T *Xarray_ = (T *) malloc( sizeof(T)*n6 * batchCount);
        T *Yarray_ = (T *) malloc( sizeof(T)*n6 * batchCount);
        T *Zarray_ = (T *) malloc( sizeof(T)*n6 * batchCount);
        T *Warray_ = (T *) malloc( sizeof(T)*n6 * batchCount);

        assert( Aarray_ != nullptr );
        assert( Xarray_ != nullptr );
        assert( Yarray_ != nullptr );
        assert( Zarray_ != nullptr );
        assert( Warray_ != nullptr );

        T *dAarray_ = (T *) myalloc( sizeof(T)*n*n*6*batchCount);
        T *dXarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );
        T *dZarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );
        T *dYarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );
        T *dWarray_ = (T *) myalloc( sizeof(T)*n6 * batchCount );

        auto Aarray = [&] (int const i, 
                           int const j, 
                           int const k, 
                           int const ibatch ) -> T& {
                return(  Aarray_[ indx4f(i,j,k,ibatch, n,n,6) ] );
        };

        auto Xarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Xarray_[ indx2f(i,ibatch,n6) ] );
        };

        auto Yarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Yarray_[ indx2f(i,ibatch,n6) ] );
        };

        auto Zarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Zarray_[ indx2f(i,ibatch,n6) ] );
        };

        auto Warray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Warray_[ indx2f(i,ibatch,n6) ] );
        };



        //  ---------------------
        //  initialize the arrays
        //  save a copy of Xarray in Z
        //  ---------------------
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
        for(int i=1; i <= n6; i++) {
              T const r1 = (i + (ibatch-1)*n6 );
              T const r2 = n6*batchCount;

              // --------------------------------
              // note Zarray is a copy of Xarray
              // --------------------------------
              Xarray(i,ibatch) = r1/r2;
              Zarray(i,ibatch) = Xarray(i,ibatch);
              Yarray(i,ibatch) = 0;
              Warray(i,ibatch) = 0;
              };
              };
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
            for(int k=1; k <= 6; k++) {
            for(int j=1; j <= n; j++) {
            for(int i=1; i <= n; i++) {
                T const r1 = i + (j-1)*n + (k-1)*n*n + (ibatch-1)*batchCount;
                T const r2 = n*n*6*batchCount;
                Aarray(i,j,k,  ibatch) = r1/r2;
            };
            };
            };
        };


        // ---------------------
        // copy from host to GPU
        // interface is host2gpu( dest, src, nbytes )
        // ---------------------
        host2gpu( dAarray_, Aarray_, sizeof(T)*n*n*6*batchCount );
        host2gpu( dXarray_, Xarray_, sizeof(T)*n6*batchCount );
        host2gpu( dYarray_, Yarray_, sizeof(T)*n6*batchCount );
        host2gpu( dZarray_, Zarray_, sizeof(T)*n6*batchCount );
        host2gpu( dWarray_, Warray_, sizeof(T)*n6*batchCount );

        auto time_start = std::chrono::steady_clock::now();
#ifdef USE_GPU
        {
        int constexpr warpsize = 32;
        int constexpr nwarps = 8;
        int constexpr nthreads = nwarps * warpsize;

        // --------------------------------------------
        // note  the input Zarray will be over-written
        // --------------------------------------------
        kronmult6_batched<T><<< batchCount, nthreads >>>( n,
                           dAarray_,
                           dZarray_,
                           dYarray_,
                           dWarray_,
                           batchCount );

        // -------------------------------------------
        // note important to wait for kernel to finish
        // -------------------------------------------
        cudaError_t istat = cudaDeviceSynchronize();
        assert( istat == cudaSuccess );
        }
#else
        {
        kronmult6_batched<T>( n,
                           dAarray_,
                           dZarray_,
                           dYarray_,
                           dWarray_,
                           batchCount );
        }
#endif
        auto time_end = std::chrono::steady_clock::now();
        auto elapsed_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        auto elapsed_time_sec = elapsed_time_ms * 0.001;

        // ------------------------------------------
        // copy from gpu to host
        // interface is gpu2host( dest, src, nbytes )
        // ------------------------------------------
        gpu2host( Yarray_, dYarray_,  sizeof(T)*n6*batchCount);



        {
          int const n7 = n*n6;
          double const giga = 1000.0*1000.0*1000.0;
          double const flops = 12.0*n7 * batchCount;
          double const gflops = flops/giga;
          double const gflops_per_sec = gflops  /elapsed_time_sec;
          if (flops > giga) {
                  std::cout << " n = " << n 
                            << " batchCount = " << batchCount
                            << " elapsed_time = " << elapsed_time_sec << " seconds "
                            << " Gflops/sec = " << gflops_per_sec
                            << "\n";
          };
        };


   T max_abserr = 0;
   if (do_check) {
        // -------------
        // check results
        // -------------
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                T const * const A1_ = &(Aarray(1,1,1,ibatch));
                T const * const A2_ = &(Aarray(1,1,2,ibatch));
                T const * const A3_ = &(Aarray(1,1,3,ibatch));
                T const * const A4_ = &(Aarray(1,1,4,ibatch));
                T const * const A5_ = &(Aarray(1,1,5,ibatch));
                T const * const A6_ = &(Aarray(1,1,6,ibatch));
                T const * const X_ = &(Xarray(1,ibatch));
                T       * const Y_ = &(Yarray(1,ibatch));

                auto X = [&] (int const i) -> T const & {
                        return( X_[ (i)-1 ]);
                };

                auto Y = [&] (int const i) -> T& {
                        return( Y_[ (i)-1 ]);
                };

                auto A1 = [&](int const i,
                              int const j) -> T const & {
                        return( A1_[ indx2f(i,j,n) ] );
                };

                auto A2 = [&](int const i,
                              int const j) -> T const & {
                        return( A2_[ indx2f(i,j,n) ] );
                };

                auto A3 = [&](int const i,
                              int const j) -> T const & {
                        return( A3_[ indx2f(i,j,n) ] );
                };

                auto A4 = [&](int const i,
                              int const j) -> T const & {
                        return( A4_[ indx2f(i,j,n) ] );
                };

                auto A5 = [&](int const i,
                              int const j) -> T const & {
                        return( A5_[ indx2f(i,j,n) ] );
                };

                auto A6 = [&](int const i,
                              int const j) -> T const & {
                        return( A6_[ indx2f(i,j,n) ] );
                };



                #pragma omp parallel for collapse(6)  reduction(max:max_abserr)
                for(int i1=1; i1 <= n; i1++) {
                for(int i2=1; i2 <= n; i2++) {
                for(int i3=1; i3 <= n; i3++) {
                for(int i4=1; i4 <= n; i4++) {
                for(int i5=1; i5 <= n; i5++) {
                for(int i6=1; i6 <= n; i6++) {

                   int const ic = 1+indx6f( i6,i5,i4,i3,i2,i1,n,n,n,n,n);
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
                      int const jc = 1+indx6f( j6,j5,j4,j3,j2,j1,n,n,n,n,n);


                      T const C_ic_jc = ((((A1(i1,j1)*A2(i2,j2))*A3(i3,j3))*A4(i4,j4))*A5(i5,j5))*A6(i6,j6);

                      T const X_jc = X(jc);

                      Y_ic += C_ic_jc * X_jc;
                   };
                   };
                   };
                   };
                   };
                   };

                   T const abs_err = std::abs( Y_ic - Y(ic) );
                   max_abserr = std::max( max_abserr, abs_err );

                   if (idebug >= 1) {
                       T const tol = 1.0/(1000.0 * 1000.0);
                       if (abs_err > tol ) {
                             std::cout << " ic = " << ic 
                                     << " Y_ic = " << Y_ic
                                     << " Y(ic) =  " << Y(ic)
                                     << " abs_err = " << abs_err << "\n";
                       };
                   };
                                    

                };
                };
                };
                };
                };
                };
       }; // end for ibatch

      };



        // -------
        // cleanup
        // -------

        myfree( dAarray_ ); dAarray_ = nullptr;
        myfree( dXarray_ ); dXarray_ = nullptr;
        myfree( dYarray_ ); dYarray_ = nullptr;
        myfree( dZarray_ ); dZarray_ = nullptr;
        myfree( dWarray_ ); dWarray_ = nullptr;

        free( Aarray_ ); Aarray_ = nullptr;
        free( Xarray_ ); Xarray_ = nullptr;
        free( Yarray_ ); Yarray_ = nullptr;
        free( Zarray_ ); Zarray_ = nullptr;
        free( Warray_ ); Warray_ = nullptr;

        return(max_abserr);

}


                      
int main() {

        int const idebug = 1;

        int batch_table[] = {1,16,128};
        int const size_batch_table = sizeof(batch_table)/sizeof(batch_table[0]);

        int n_table[] = {1, 2,3, 4 };
        int const size_n_table = sizeof(n_table)/sizeof(n_table[0]);


        int nerrors = 0;

        for (int ibatch_table=0; ibatch_table < size_batch_table; ibatch_table++) {
        for (int in_table = 0;  in_table < size_n_table; in_table++) {
                int const n = n_table[in_table];
                int const batchCount = batch_table[ibatch_table];

                double const max_abserr =  test_kronmult6_batched<double>(n, batchCount, idebug );
                double const tol = 1.0/(1000.0 * 1000.0);
                bool const isok = (max_abserr <= tol);
                if (!isok) {
                        nerrors += 1;
                };

                if ((idebug >= 1) || (!isok)) {
                        std::cout << " n = " << n << " batchCount = " << batchCount
                                  << " max_abserr= " << max_abserr << "\n";
                };
        };
        };


        if (nerrors == 0) {
                std::cout << "ALL PASSED" << "\n";
        }
        else {
                std::cout << "There are " << nerrors << " errors" << "\n";
        };

        if (nerrors == 0) {
               // ---------------------
               // try performance test
               // ---------------------
               int const batchCount = 256;
               bool const do_check = 0;
               int const idebug = 0;


               for(int n=7; n <= 8; n++) {
                test_kronmult6_batched<double>(n, batchCount, idebug, do_check );
               };
        };




  return(0);
}


                     


