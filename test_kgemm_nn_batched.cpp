#include <iostream>
#include <cassert>

#include "kgemm_nn_batched.hpp"

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

int main()
{
        // ----------------------------------------
        // simple program to test kgemm_nn_batched
        // ----------------------------------------

        int constexpr n = 16;
        int constexpr  batchCount = 4;

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
                double * const A_ = new double[n*n];
                double * const B_ = new double[n*n];
                double * const C_ = new double[n*n];

                assert( A_ != 0);
                assert( B_ != 0);
                assert( C_ != 0);

                Aarray_[ibatch] = A_;
                Barray_[ibatch] = B_;
                Carray_[ibatch] = C_;

        };


        int const ldA = n;
        int const ldB = n;
        int const ldC = n;
#define A(i,j)  A_[ indx2f(i,j,ldA) ]
#define B(i,j)  B_[ indx2f(i,j,ldB) ]
#define C(i,j)  C_[ indx2f(i,j,ldC) ]

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
             double *A_ = Aarray_[ibatch];
             double *B_ = Barray_[ibatch];
             double *C_ = Carray_[ibatch];
             for(int j=1; j <= n; j++) {
             for(int i=1; i <= n; i++) {
                A(i,j) = 1.0 + i + j;
                B(i,j) = 1.0/(1.0 + i + j);
                C(i,j) = 0;
             };
             };
        };

        // --------------------------
        // allocate storage on device
        // --------------------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {

                const size_t nbytes = sizeof(double)*n*n;
                double *dA = nullptr;
                cudaError_t istat_dA = cudaMalloc( &dA, nbytes );
                assert( istat_dA == cudaSuccess );

                double *dB = nullptr;
                cudaError_t istat_dB = cudaMalloc( &dB, nbytes );
                assert( istat_dB == cudaSuccess );

                double *dC = nullptr;
                cudaError_t istat_dC = cudaMalloc( &dC, nbytes );
                assert( istat_dC == cudaSuccess );

                hdAarray_[ibatch] = dA;
                hdBarray_[ibatch] = dB;
                hdCarray_[ibatch] = dC;
        };

        // ----------------------
        // copy matices to device
        // ----------------------

        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                {
                size_t nbytes = sizeof(double)*n*n;
                void * const dest = hdAarray_[ibatch];
                void const * const src =  Aarray_[ibatch];
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy(  dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
                };

                {
                size_t nbytes = sizeof(double)*n*n;
                void * const dest = hdBarray_[ibatch];
                void const * const src =  Barray_[ibatch];
                cudaMemcpyKind const kinddir = cudaMemcpyHostToDevice;
                cudaError_t const istat_cpy = cudaMemcpy(  dest, src, nbytes, kinddir );
                assert( istat_cpy == cudaSuccess );
                };

                {
                size_t nbytes = sizeof(double)*n*n;
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

         cudaError_t istat_ldB = cudaMalloc( &dldBarray_, nbytes );
         assert( istat_ldB == cudaSuccess );

         cudaError_t istat_ldC = cudaMalloc( &dldCarray_, nbytes );
         assert( istat_ldC == cudaSuccess );
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

        int const mm = n;
        int const nn = n;
        int const kk = n;
        {
        double const alpha = 1.0;
        double const beta = 0.0;

        dim3 grid(batchCount,1,1);
        dim3 block(16,16,1);
        kgemm_nn_batched<double><<< grid, block >>>( mm,nn,kk, 
                          alpha,
                          ddAarray_, dldAarray_,
                          ddBarray_, dldBarray_,
                          beta, 
                          ddCarray_, dldCarray_,
                          batchCount);
        }


        // -------------
        // check results
        // -------------
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
                size_t const nbytes = sizeof(double) * n * n;
                void * const dest = Carray_[ibatch];
                void * const src = hdCarray_[ibatch];
                cudaMemcpyKind kinddir = cudaMemcpyDeviceToHost;
                cudaError_t istat_cpy = cudaMemcpy( dest, src, nbytes, kinddir );
        };


        double max_abserr = 0;
        for(int ibatch=0; ibatch < batchCount; ibatch++) {
              double const * const A_ = Aarray_[ibatch];
              double const * const B_ = Barray_[ibatch];
              double const * const C_ = Carray_[ibatch];

              for(int j=1; j <= nn; j++) {
              for(int i=1; i <= mm; i++) {
                      double cij = 0;
                      for(int k=1; k <= kk; k++) {
                             cij += A(i,k) * B(k,j);
                      };
                      double const abserr = ABS( cij  - C(i,j) );
                      max_abserr = MAX( max_abserr, abserr );
              };
              };
        }; 

        std::cout << "n = " << n << " batchCount = " << batchCount << "\n";
        std::cout << "max_abserr = " << max_abserr << "\n";

        return(0);
}












