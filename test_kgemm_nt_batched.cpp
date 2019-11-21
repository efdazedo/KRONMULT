#include <iostream>
#include <cassert>
#include <chrono>
#include <unistd.h>

#include "kgemm_nt_batched.hpp"

#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef indx2f
#define indx2f(i,j,ld)  (((i)-1) + ((j)-1)*(ld))
#endif

#ifndef MAX
#define MAX(x,y)  (  ((x) > (y))?(x):(y) )
#endif

#ifndef MIN
#define MIN(x,y)  (  ((x) < (y))?(x):(y) )
#endif

#ifndef ABS
#define ABS(x) (((x) >= 0) ? (x) : (-(x)) )
#endif

double test_kgemm_nt_batched( int const mm, 
                              int const nn, 
                              int const kk, 
                              int const batchCount, 
                              int const idebug = 1)
{
        // ----------------------------------------
        // simple program to test kgemm_nt_batched
        // ----------------------------------------

        double const alpha = 1.3;
        double const beta = 1.2;


        int const nrowA = mm; 
        int const ncolA = kk; 

        int const nrowB = nn; 
        int const ncolB = kk; 

        int const nrowC = mm; 
        int const ncolC = nn; 



        int const wsize = 32;
        int const ldA = wsize * (( nrowA + (wsize-1))/wsize );
        int const ldB = wsize * (( nrowB + (wsize-1))/wsize );
        int const ldC = wsize * (( nrowC + (wsize-1))/wsize );

        double *Aarray_[batchCount];
        double *Barray_[batchCount];
        double *Carray_[batchCount];

        // -------------
        // device arrays
        // -------------
        double *hdAarray_[batchCount];
        double *hdBarray_[batchCount];
        double *hdCarray_[batchCount];

        double **ddAarray_ = nullptr;
        double **ddBarray_ = nullptr;
        double **ddCarray_ = nullptr;

        {
                size_t nbytes = sizeof(double *) * batchCount;
                cudaError_t istat_ddAarray = cudaMalloc(  &ddAarray_, nbytes );
                assert(istat_ddAarray == cudaSuccess );

                cudaError_t istat_ddBarray = cudaMalloc(  &ddBarray_, nbytes );
                assert(istat_ddBarray == cudaSuccess );

                cudaError_t istat_ddCarray = cudaMalloc(  &ddCarray_, nbytes );
                assert(istat_ddCarray == cudaSuccess );
        };

        

        // ----------------
        // initialize array
        // ----------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                double * const A_ = new double[ldA*ncolA];
                double * const B_ = new double[ldB*ncolB];
                double * const C_ = new double[ldC*ncolC];

                assert( A_ != nullptr);
                assert( B_ != nullptr);
                assert( C_ != nullptr);

                Aarray_[ibatch] = A_;
                Barray_[ibatch] = B_;
                Carray_[ibatch] = C_;

        };


#define A(i,j)  A_[ indx2f(i,j,ldA) ]
#define B(i,j)  B_[ indx2f(i,j,ldB) ]
#define C(i,j)  C_[ indx2f(i,j,ldC) ]

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
             double *A_ = Aarray_[ibatch];
             double *B_ = Barray_[ibatch];
             double *C_ = Carray_[ibatch];
             for(int j=1; j <= ncolA; j++) {
             for(int i=1; i <= nrowA; i++) {
                A(i,j) = 1.0 + i + j + ibatch;
             };
             };


             for(int j=1; j <= ncolB; j++) {
             for(int i=1; i <= nrowB; i++) {
                B(i,j) = 1.0 /(1.0 + i + j + ibatch);
             };
             };

             for(int j=1; j <= ncolC; j++) {
             for(int i=1; i <= nrowC; i++) {
                C(i,j) = 1;
             };
             };
        };

        // --------------------------
        // allocate storage on device
        // --------------------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {

                size_t const nbytes_A = sizeof(double)*ldA*ncolA;
                double *dA = nullptr;
                cudaError_t istat_dA = cudaMalloc( &dA, nbytes_A );
                assert( istat_dA == cudaSuccess );

                size_t const nbytes_B = sizeof(double)*ldB*ncolB;
                double *dB = nullptr;
                cudaError_t istat_dB = cudaMalloc( &dB, nbytes_B );
                assert( istat_dB == cudaSuccess );

                size_t const nbytes_C = sizeof(double)*ldC*ncolC;
                double *dC = nullptr;
                cudaError_t istat_dC = cudaMalloc( &dC, nbytes_C );
                assert( istat_dC == cudaSuccess );

                assert( dA != nullptr );
                assert( dB != nullptr );
                assert( dC != nullptr );

                hdAarray_[ibatch] = dA;
                hdBarray_[ibatch] = dB;
                hdCarray_[ibatch] = dC;
        };

        // ----------------------
        // copy matices to device
        // ----------------------

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                {
                size_t nbytes = sizeof(double)*ldA*ncolA;
                void * const dest = hdAarray_[ibatch];
                void const * const src =  Aarray_[ibatch];
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy(  dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
                };

                {
                size_t nbytes = sizeof(double)*ldB*ncolB;
                void * const dest = hdBarray_[ibatch];
                void const * const src =  Barray_[ibatch];
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy(  dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
                };

                {
                size_t nbytes = sizeof(double)*ldC*ncolC;
                void * const dest = hdCarray_[ibatch];
                void const * const src =  Carray_[ibatch];
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy(  dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
                };
        };

        // -----------------------
        // copy pointers to device
        // -----------------------
        {
                size_t nbytes = sizeof( double *) * batchCount;
                void *dest = ddAarray_;
                void *src = &(hdAarray_[0]);
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
        }
        {
                size_t nbytes = sizeof( double *) * batchCount;
                void *dest = ddBarray_;
                void *src = &(hdBarray_[0]);
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
        }

        {
                size_t nbytes = sizeof( double *) * batchCount;
                void *dest = ddCarray_;
                void *src = &(hdCarray_[0]);
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
        }


        // --------------------------
        // setup ldA, ldB, ldC arrays
        // --------------------------
        int ldAarray_[batchCount];
        int ldBarray_[batchCount];
        int ldCarray_[batchCount];
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                ldAarray_[ibatch] = ldA;
                ldBarray_[ibatch] = ldB;
                ldCarray_[ibatch] = ldC;
        };

        int *dldAarray_ = nullptr;
        int *dldBarray_ = nullptr;
        int *dldCarray_ = nullptr;

        {
         size_t nbytes = sizeof(int) * batchCount;
         cudaError_t istat_ldA = cudaMalloc( &dldAarray_, nbytes );
         assert( istat_ldA == cudaSuccess );
         assert( dldAarray_ != nullptr);

         cudaError_t istat_ldB = cudaMalloc( &dldBarray_, nbytes );
         assert( istat_ldB == cudaSuccess );
         assert( dldBarray_ != nullptr);

         cudaError_t istat_ldC = cudaMalloc( &dldCarray_, nbytes );
         assert( istat_ldC == cudaSuccess );
         assert( dldCarray_ != nullptr);
        }

        // -------------------------------------------
        // copy array  ldAarray_, ldBarray_, ldCarray_
        // -------------------------------------------

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldAarray_[0]);
                void *dest = dldAarray_;
                cudaMemcpyKind kinddir = cudaMemcpyHostToDevice;
                cudaError_t istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
        }

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldBarray_[0]);
                void *dest = dldBarray_;
                cudaMemcpyKind kinddir = cudaMemcpyHostToDevice;
                cudaError_t istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
        }

        {
                size_t nbytes = sizeof(int) * batchCount;
                void *src = &(ldCarray_[0]);
                void *dest = dldCarray_;
                cudaMemcpyKind kinddir = cudaMemcpyHostToDevice;
                cudaError_t istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
        }

        {

        dim3 grid(batchCount,1,1);
        dim3 block(16,16,1);

        cudaError_t istat_sync_start = cudaDeviceSynchronize();
        assert( istat_sync_start == cudaSuccess );

        auto time_start = std::chrono::steady_clock::now();

        kgemm_nt_batched<double><<< grid, block >>>( mm,nn,kk, 
                          alpha,
                          ddAarray_, dldAarray_,
                          ddBarray_, dldBarray_,
                          beta, 
                          ddCarray_, dldCarray_,
                          batchCount);

        cudaError_t istat_sync_end = cudaDeviceSynchronize();
        assert( istat_sync_end == cudaSuccess );

        auto time_end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end- time_start).count();

        double elapsed_time_in_sec = elapsed_time * 0.001;
        double flops = (2.0*mm*nn)*kk*batchCount;
        double gflops_per_sec = flops/(1000.0*1000.0*1000.0) / elapsed_time_in_sec;

        if (idebug >= 2) {
          std::cout << "elapsed time is " << elapsed_time_in_sec << " seconds " 
                  << gflops_per_sec << " Gflops/s" << "\n";
          };

        }


        // -------------
        // check results
        // -------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t const nbytes = sizeof(double) * ldC*ncolC;
                void * const dest = Carray_[ibatch];
                void * const src = hdCarray_[ibatch];
                cudaMemcpyKind kinddir = cudaMemcpyDeviceToHost;
                cudaError_t istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
        };

        {
        cudaError_t istat_sync = cudaDeviceSynchronize();
        assert( istat_sync == cudaSuccess );
        }

        double max_abserr = 0;
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
              double const * const A_ = Aarray_[ibatch];
              double const * const B_ = Barray_[ibatch];
              double const * const C_ = Carray_[ibatch];

              double const cij0 = 1;

              for(int j=1; j <= nn; j++) {
              for(int i=1; i <= mm; i++) {
                      double cij = 0;
                      for(int k=1; k <= kk; k++) {
                              // ---------------------------
                              // note   C = A * transpose(B)
                              // ---------------------------
                             cij += A(i,k) * B(j,k); 
                      };
                      cij = alpha * cij + beta * cij0;

                      double const abserr = ABS( cij  - C(i,j) );
                      max_abserr = MAX( max_abserr, abserr );
              };
              };
        }; 

        if (idebug >= 1) {
          std::cout << "max_abserr = " << max_abserr << "\n";
        };

        // --------
        // clean up
        // --------

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                delete [] Aarray_[ibatch];
                delete [] Barray_[ibatch];
                delete [] Carray_[ibatch];
        };

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                {
                        cudaError_t istat = cudaFree( hdAarray_[ibatch] );
                        assert( istat == cudaSuccess );
                }
                {
                        cudaError_t istat = cudaFree( hdBarray_[ibatch] );
                        assert( istat == cudaSuccess );
                }
                {
                        cudaError_t istat = cudaFree( hdCarray_[ibatch] );
                        assert( istat == cudaSuccess );
                }
             };

        {
                cudaError_t istat_ddA = cudaFree( ddAarray_ );
                assert( istat_ddA == cudaSuccess );

                cudaError_t istat_ddB = cudaFree( ddBarray_ );
                assert( istat_ddB == cudaSuccess );

                cudaError_t istat_ddC = cudaFree( ddCarray_ );
                assert( istat_ddC == cudaSuccess );


                cudaError_t istat_ldA = cudaFree( dldAarray_ );
                assert( istat_ldA == cudaSuccess );

                cudaError_t istat_ldB = cudaFree( dldBarray_ );
                assert( istat_ldB == cudaSuccess );

                cudaError_t istat_ldC = cudaFree( dldCarray_ );
                assert( istat_ldC == cudaSuccess );


        }





        return(max_abserr);
}

int main()
{
        int const idebug = 1;
        int const inc = 7;
        int const kk_max = 40;
        int const mm_max = 40;
        int const nn_max = 40;
        int const batchCount_max = 2*inc + 1;
        double const tol = 1.0/(1000.0*1000.0);

        int nerrors = 0;
        for(int batchCount=1; batchCount <= batchCount_max; batchCount += inc) {
        for(int kk=1; kk <= kk_max; kk += inc) {
        for(int nn=1; nn <= nn_max; nn += inc) {
        for(int mm=1; mm <= mm_max; mm += inc) {
                double const max_abserr = test_kgemm_nt_batched(mm,nn,kk,batchCount,idebug);
                double const isok = (max_abserr < tol);

                if (!isok) {
                        nerrors += 1;
                };

                if ((!isok) || (idebug >= 2)) {
                        std::cout << " mm = " << mm 
                                  << " nn = " << nn
                                  << " kk = " << kk
                                  << " batchCount = " << batchCount 
                                  << " max_abserr = " << max_abserr
                                  << "\n";
                };
        };
        };
        };
        };

        if (nerrors == 0) {
                std::cout << "ALL PASSED" << "\n";
        }
        else {
                std::cout << "There are " << nerrors << " errors " << "\n";
        };
        return(0);
}






