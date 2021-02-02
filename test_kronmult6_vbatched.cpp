#include <iostream>
#include <cassert>
#include <chrono>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "kroncommon.hpp"
#include "kronmult6_vbatched.hpp"
#include "kronmult5_vbatched.hpp"
#include "kronmult4_vbatched.hpp"
#include "kronmult3_vbatched.hpp"
#include "kronmult2_vbatched.hpp"
#include "kronmult1_vbatched.hpp"




static inline
void host2gpu( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        hipError_t istat = hipMemcpy( dest, 
                                        src, 
                                        nbytes,  
                                        hipMemcpyHostToDevice );
        assert( istat == hipSuccess );
#else
        memcpy( dest, src, nbytes );
#endif
}

static inline
void gpu2host( void *dest, void *src, size_t nbytes )
{
#ifdef USE_GPU
        hipError_t istat = hipMemcpy( dest,
                                        src,
                                        nbytes,
                                        hipMemcpyDeviceToHost);
        assert( istat == hipSuccess );
#else
        memcpy( dest, src, nbytes );
#endif

}

static inline
void *myalloc( size_t nbytes ) {
              void *devPtr = nullptr;
#ifdef USE_GPU
              hipError_t istat = hipMalloc( &devPtr, nbytes );
              assert( istat == hipSuccess );
#else
              devPtr = malloc( nbytes );
#endif
              assert( devPtr != nullptr );
              return(devPtr);
}

static inline
void myfree( void * devPtr ) {
#ifdef USE_GPU
                hipError_t istat = hipFree( devPtr);
                assert( istat == hipSuccess );
#else
                free( devPtr );
#endif
}
     

template<typename T, typename Tc=double>
double test_kronmult_vbatched(  int const idim,
                          int const m_array[], 
			  int const n_array[],
			  int const batchCount, 
                          int const idebug = 1, 
                          bool const do_check  = true,
                          bool const use_overlap_in_Y = false,
	                  size_t Wcapacity_bytes_in = 0	)
        
{


	int const m1 = (idim >= 1) ? m_array[0] : 1; int const n1 = (idim >= 1) ? n_array[0] : 1;
	int const m2 = (idim >= 2) ? m_array[1] : 1; int const n2 = (idim >= 2) ? n_array[1] : 1;
	int const m3 = (idim >= 3) ? m_array[2] : 1; int const n3 = (idim >= 3) ? n_array[2] : 1;
	int const m4 = (idim >= 4) ? m_array[3] : 1; int const n4 = (idim >= 4) ? n_array[3] : 1;
	int const m5 = (idim >= 5) ? m_array[4] : 1; int const n5 = (idim >= 5) ? n_array[4] : 1;
	int const m6 = (idim >= 6) ? m_array[5] : 1; int const n6 = (idim >= 6) ? n_array[5] : 1;



        // -------------------------
        // Aarray is (lda,n,idim,batchCount)
	// Aparray is (idim,batchCount)
        // Xarray is (n1*n2..*n6 by batchCount)
        // Yarray is (m1*m2..*m6 by batchCount)
        // Zarray is (m1*m2..*m6 by batchCount)
        // Warray is (Wsize)
        // ----------------------------

        size_t const Xsize = n1*n2*n3*n4*n5*n6;
	size_t const Zsize = Xsize;
	size_t const Ysize = m1*m2*m3*m4*m5*m6;

	// ------------------
	// compute n to be max
	// just for simplicity
	// ------------------
	int n = 0;
	{
	  int nmax = 0;
	  int mmax = 0;
	  for(int i=0; i < idim; i++) { 
		nmax = std::max(nmax, n_array[i] );
		mmax = std::max(mmax, m_array[i]);
	  };
	  n = std::max( nmax, mmax );
	};
	int const lda = n;


	int const ld1 = m1;
	int const ld2 = m2;
	int const ld3 = m3;
	int const ld4 = m4;
	int const ld5 = m5;
	int const ld6 = m6;

	size_t const Aarray_nbytes = sizeof(T)*(lda*n)*idim*batchCount;
	size_t const Aparray_nbytes = sizeof(T*) * idim * batchCount;

	size_t const Wcapacity_bytes_default = 1024 * 1024 * 1024;
	size_t const Wcapacity_bytes = (Wcapacity_bytes_in == 0) ? Wcapacity_bytes_default: Wcapacity_bytes_in;

        T *Aarray_   = (T *)  malloc( Aarray_nbytes );
        T **Aparray_ = (T **) malloc( Aparray_nbytes );

        T *Xarray_ = (T *) malloc( sizeof(T)*Xsize * batchCount);
        T *Zarray_ = (T *) malloc( sizeof(T)*Zsize * batchCount);

        T *Yarray_ = (T *) malloc( sizeof(T)*Ysize * batchCount);
        T *Y2array_ = (T *) malloc( sizeof(T)*Ysize * batchCount);

        T *Warray_ = (T *) malloc( Wcapacity_bytes );

        assert( Aarray_ != nullptr );
        assert( Aparray_ != nullptr );

        assert( Xarray_ != nullptr );
        assert( Yarray_ != nullptr );
        assert( Y2array_ != nullptr );

	assert( Warray_ != nullptr );
        assert( Zarray_ != nullptr );

	memset( Aarray_,0,Aarray_nbytes);
	memset( Aparray_,0,Aparray_nbytes);

	memset( Xarray_,0,sizeof(T)*Xsize * batchCount);
	memset( Zarray_,0,sizeof(T)*Zsize * batchCount);

	memset( Yarray_,0,sizeof(T)*Ysize * batchCount);
	memset( Y2array_,0,sizeof(T)*Ysize * batchCount);

	memset( Warray_,0,Wcapacity_bytes);



        T *dAarray_   = (T *)  myalloc( Aarray_nbytes );
	T **dAparray_ = (T **) myalloc( Aparray_nbytes );


        T *dXarray_ = (T *) myalloc( sizeof(T)*Xsize * batchCount );
        T *dZarray_ = (T *) myalloc( sizeof(T)*Zsize * batchCount);

        T *dYarray_ = (T *) myalloc( sizeof(T)*Ysize * batchCount );
        T *dWarray_ = (T *) myalloc( Wcapacity_bytes );

        assert( dAarray_  != nullptr );
        assert( dAparray_ != nullptr );

        assert( dXarray_ != nullptr );
        assert( dYarray_ != nullptr );
        assert( dZarray_ != nullptr );
        assert( dWarray_ != nullptr );

        T** pdXarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdYarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdZarray_ = (T**) malloc( sizeof(T*) * batchCount );
        T** pdWarray_ = (T**) malloc( sizeof(T*) * batchCount );

	assert( pdXarray_ != nullptr );
	assert( pdYarray_ != nullptr );
	assert( pdZarray_ != nullptr );
	assert( pdWarray_ != nullptr );

	memset( pdXarray_,0,sizeof(T*) * batchCount);
	memset( pdYarray_,0,sizeof(T*) * batchCount);
	memset( pdZarray_,0,sizeof(T*) * batchCount);
	memset( pdWarray_,0,sizeof(T*) * batchCount);

        T** dpdXarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdZarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdYarray_ = (T**) myalloc( sizeof(T*) * batchCount );
        T** dpdWarray_ = (T**) myalloc( sizeof(T*) * batchCount );

        assert( dpdXarray_ != nullptr );
        assert( dpdYarray_ != nullptr );
        assert( dpdZarray_ != nullptr );
        assert( dpdWarray_ != nullptr );



        auto dAarray = [&] (int const i, 
                           int const j, 
                           int const k, 
                           int const ibatch ) -> T& {
                return(  dAarray_[ indx4f(i,j,k,ibatch, lda,n,idim) ] );
        };

        auto Aarray = [&] (int const i, 
                           int const j, 
                           int const k, 
                           int const ibatch ) -> T& {
                return(  Aarray_[ indx4f(i,j,k,ibatch, lda,n,idim) ] );
        };

	auto Aparray = [&] (int const i,
			    int const ibatch ) -> T* & {
		return( Aparray_[ indx2f(i,ibatch,idim) ] );
	};

        auto Xarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Xarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto Yarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Yarray_[ indx2f(i,ibatch,Ysize) ] );
        };

        auto Y2array = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Y2array_[ indx2f(i,ibatch,Ysize) ] );
        };

        auto Zarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Zarray_[ indx2f(i,ibatch,Zsize) ] );
        };

#if (0)
        auto Warray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( Warray_[ indx2f(i,ibatch,Wsize) ] );
        };
#endif

        auto dXarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dXarray_[ indx2f(i,ibatch,Xsize) ] );
        };

        auto dYarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dYarray_[ indx2f(i,ibatch,Ysize) ] );
        };

        auto dZarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dZarray_[ indx2f(i,ibatch,Zsize) ] );
        };

        auto dWarray = [&] (int const i, 
                           int const ibatch) -> T& {
                return( dWarray_[ indx2f(i,ibatch,Xsize) ] );
        };


        //  ---------------------
        //  initialize the arrays
        //  save a copy of Xarray in Z
        //  ---------------------
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
          for(int i=1; i <= Xsize; i++) {
              T const r1 = (i + (ibatch-1)*Xsize );
              T const r2 = Xsize*batchCount;

              // --------------------------------
              // note Zarray is a copy of Xarray
              // --------------------------------
              Xarray(i,ibatch) = r1/r2;
              Zarray(i,ibatch) = Xarray(i,ibatch);
              };
	   for(int i=1; i <= Ysize; i++) {
              Yarray(i,ibatch) = 0;
	   };
         }; // for ibatch
#ifdef _OPENMP
        #pragma omp parallel for 
#endif
        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
            for(int k=1; k <= idim; k++) {
            for(int j=1; j <= n; j++) {
            for(int i=1; i <= n; i++) {
                T const r1 = i + (j-1)*n + (k-1)*n*n + (ibatch-1)*batchCount;
                T const r2 = n*n*idim*batchCount;
                Aarray(i,j,k,  ibatch) = r1/r2;
            };
            };
            };
        };

#ifdef _OPENMP
        #pragma omp parallel for
#endif
	for(int ibatch=1; ibatch <= batchCount; ibatch++) {
	   for(int k=1; k <= idim; k++) {
		Aparray(k,ibatch) = &(dAarray(1,1,k,ibatch));
	   };
	};


        // ---------------------
        // copy from host to GPU
        // interface is host2gpu( dest, src, nbytes )
        // ---------------------
        host2gpu( dAarray_,  Aarray_,  Aarray_nbytes );
        host2gpu( dAparray_, Aparray_, Aparray_nbytes );

        host2gpu( dXarray_, Xarray_, sizeof(T)*Xsize*batchCount );
        host2gpu( dZarray_, Zarray_, sizeof(T)*Zsize*batchCount );
        host2gpu( dYarray_, Yarray_, sizeof(T)*Ysize*batchCount );

	memset( Warray_, 0, Wcapacity_bytes );
        host2gpu( dWarray_, Warray_, Wcapacity_bytes );

        for(int ibatch=1; ibatch <= batchCount;  ibatch++) {
                pdXarray_[ (ibatch-1) ] = &(dXarray(1,ibatch));
                
                if (use_overlap_in_Y) {
                  pdYarray_[ (ibatch-1) ] = &(dYarray(1,1));
                }
                else {
                  pdYarray_[ (ibatch-1) ] = &(dYarray(1,ibatch));
                };

                pdZarray_[ (ibatch-1) ] = &(dZarray(1,ibatch));
                pdWarray_[ (ibatch-1) ] = &(dWarray(1,ibatch));
        };

        host2gpu( dpdXarray_, pdXarray_, sizeof(T*)*batchCount );
        host2gpu( dpdYarray_, pdYarray_, sizeof(T*)*batchCount );
        host2gpu( dpdZarray_, pdZarray_, sizeof(T*)*batchCount );
        host2gpu( dpdWarray_, pdWarray_, sizeof(T*)*batchCount );



        auto time_start = std::chrono::steady_clock::now();

	int const m_[6] = {m1,m2,m3,m4,m5,m6};
	int const n_[6] = {n1,n2,n3,n4,n5,n6};

#ifdef USE_GPU
        {
        int constexpr warpsize = WARPSIZE;
        int const nwarps = 2;
        int const nthreads = nwarps * warpsize;

        // --------------------------------------------
        // note  the input Zarray will be over-written
        // --------------------------------------------
        switch(idim) { 
        case 1:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,1>), dim3(batchCount), dim3(nthreads ), 0, 0,  
			   m_, n_, 
			   dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_, 
			   Wcapacity_bytes,
                           batchCount );
            break;
        case 2:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,2>), dim3(batchCount), dim3(nthreads ), 0, 0,  
                           m_, n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;
        case 3:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,3>), dim3(batchCount), dim3(nthreads ), 0, 0,  
                           m_, n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;
        case 4:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,4>), dim3(batchCount), dim3(nthreads ), 0, 0,  
                           m_, n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;
        case 5:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,5>), dim3(batchCount), dim3(nthreads ), 0, 0,  
                           m_, n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;
        case 6:  hipLaunchKernelGGL(HIP_KERNEL_NAME(kronmult_vbatched<T,6>), dim3(batchCount), dim3(nthreads ), 0, 0,  
                           m_, n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;
         default: 
            assert( false );
        };

        // -------------------------------------------
        // note important to wait for kernel to finish
        // -------------------------------------------
        hipError_t istat = hipDeviceSynchronize();
        assert( istat == hipSuccess );
        }
#else

        {

        // --------------------------------------------
        // note  the input Zarray will be over-written
        // --------------------------------------------
        switch(idim) { 
        case 1:  kronmult_vbatched<T,1>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;

        case 2:  kronmult_vbatched<T,2>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;

	case 3: kronmult_vbatched<T,3>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;

	case 4: kronmult_vbatched<T,4>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;

	case 5: kronmult_vbatched<T,5>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;


	case 6: kronmult_vbatched<T,6>( m_,n_,
                           dAparray_, 
                           dpdZarray_,
                           dpdYarray_,
                           dWarray_,
			   Wcapacity_bytes,
                           batchCount );
            break;


         default: 
            assert( false );
        };

     }




#endif
        auto time_end = std::chrono::steady_clock::now();
        auto elapsed_time_us = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count();
        auto elapsed_time_sec = elapsed_time_us * 0.001 * 0.001;

        // ------------------------------------------
        // copy from gpu to host
        // interface is gpu2host( dest, src, nbytes )
        // ------------------------------------------
        gpu2host( Yarray_, dYarray_,  sizeof(T)*Ysize*batchCount);




        {
          double const giga = 1000.0*1000.0*1000.0;
          double const flops = kron_flops(idim,m_array,n_array) * batchCount;
          double const gflops = flops/giga;
          double const gflops_per_sec = gflops  /elapsed_time_sec;
          if (flops > 0.01 * giga) {
                  std::cout << " idim = " << idim
                            << " n = " << n 
                            << " batchCount = " << batchCount
                            << " elapsed_time = " << elapsed_time_sec << " seconds "
                            << " Gflops/sec = " << gflops_per_sec
                            << "\n";
          };
        };


   Tc max_abserr = 0;
   Tc max_relerr = 0;
   if (do_check) {
        // -------------
        // check results
        // -------------

        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                T const * const A1_ = (idim >= 1) ? &(Aarray(1,1,1,ibatch)) : nullptr;
                T const * const A2_ = (idim >= 2) ? &(Aarray(1,1,2,ibatch)) : nullptr;
                T const * const A3_ = (idim >= 3) ? &(Aarray(1,1,3,ibatch)) : nullptr;
                T const * const A4_ = (idim >= 4) ? &(Aarray(1,1,4,ibatch)) : nullptr;
                T const * const A5_ = (idim >= 5) ? &(Aarray(1,1,5,ibatch)) : nullptr;
                T const * const A6_ = (idim >= 6) ? &(Aarray(1,1,6,ibatch)) : nullptr;

                T const * const X_ = &(Xarray(1,ibatch));


                auto X = [&] (int const i) -> T const & {
                        return( X_[ (i)-1 ]);
                };

                auto A1 = [&](int const i,
                              int const j) -> T const & {
                        return( A1_[ indx2f(i,j,ld1) ] );
                };

                auto A2 = [&](int const i,
                              int const j) -> T const & {
                        return( A2_[ indx2f(i,j,ld2) ] );
                };

                auto A3 = [&](int const i,
                              int const j) -> T const & {
                        return( A3_[ indx2f(i,j,ld3) ] );
                };

                auto A4 = [&](int const i,
                              int const j) -> T const & {
                        return( A4_[ indx2f(i,j,ld4) ] );
                };

                auto A5 = [&](int const i,
                              int const j) -> T const & {
                        return( A5_[ indx2f(i,j,ld5) ] );
                };

                auto A6 = [&](int const i,
                              int const j) -> T const & {
                        return( A6_[ indx2f(i,j,ld6) ] );
                };


                int const max_i1 = (idim >= 1) ? m1 : 1;
                int const max_i2 = (idim >= 2) ? m2 : 1;
                int const max_i3 = (idim >= 3) ? m3 : 1;
                int const max_i4 = (idim >= 4) ? m4 : 1;
                int const max_i5 = (idim >= 5) ? m5 : 1;
                int const max_i6 = (idim >= 6) ? m6 : 1;

                int const max_j1 = (idim >= 1) ? n1 : 1;
                int const max_j2 = (idim >= 2) ? n2 : 1;
                int const max_j3 = (idim >= 3) ? n3 : 1;
                int const max_j4 = (idim >= 4) ? n4 : 1;
                int const max_j5 = (idim >= 5) ? n5 : 1;
                int const max_j6 = (idim >= 6) ? n6 : 1;

#ifdef _OPENMP
                #pragma omp parallel for collapse(6)  
#endif
                for(int i1=1; i1 <= max_i1; i1++) 
                for(int i2=1; i2 <= max_i2; i2++) 
                for(int i3=1; i3 <= max_i3; i3++) 
                for(int i4=1; i4 <= max_i4; i4++) 
                for(int i5=1; i5 <= max_i5; i5++) 
                for(int i6=1; i6 <= max_i6; i6++) {

                   int const ic = 1+indx6f( i6,i5,i4,i3,i2,i1,
                                            max_i6, max_i5, max_i4, 
                                            max_i3, max_i2 );
                   Tc Y_ic = 0;


                   for(int j1=1; j1 <= max_j1; j1++) {
                   for(int j2=1; j2 <= max_j2; j2++) {
                   for(int j3=1; j3 <= max_j3; j3++) {
                   for(int j4=1; j4 <= max_j4; j4++) {
                   for(int j5=1; j5 <= max_j5; j5++) {
                   for(int j6=1; j6 <= max_j6; j6++) {

                      // -------------------------------
                      // note last index i6 goes fastest
                      // -------------------------------
                      int const jc = 1+indx6f( j6,j5,j4,j3,j2,j1,
                                               max_j6, max_j5, max_j4,
                                               max_j3, max_j2 );


                      Tc C_ic_jc =  1;
                      C_ic_jc *= (idim >= 1) ? A1(i1,j1) : 1;
                      C_ic_jc *= (idim >= 2) ? A2(i2,j2) : 1;
                      C_ic_jc *= (idim >= 3) ? A3(i3,j3) : 1;
                      C_ic_jc *= (idim >= 4) ? A4(i4,j4) : 1;
                      C_ic_jc *= (idim >= 5) ? A5(i5,j5) : 1;
                      C_ic_jc *= (idim >= 6) ? A6(i6,j6) : 1;




                      Tc const X_jc = X(jc);

                      Y_ic += C_ic_jc * X_jc;
                   };
                   };
                   };
                   };
                   };
                   };

                   Y2array(ic,ibatch) = Y_ic;
                                    

                
                
                
                
                
                };
          }; // end for ibatch

                int const max_ic = n1*n2*n3*n4*n5*n6;
                for(int ic=1; ic <= max_ic; ic++) { 
                   Tc Y_ic = 0;
                   Tc Yval = 0;
                   Tc abs_err = 0;
                   Tc rel_err = 0;


                   if (use_overlap_in_Y) {
                        for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                                Yval += Y2array(ic,ibatch);
                        };
                        abs_err = std::abs( Yval - Yarray(ic,1) );
		        rel_err = abs_err/(1+std::max( std::abs(Yval),std::abs(Y_ic) ));
                   }
                   else {
                       for(int ibatch=1; ibatch <= batchCount; ibatch++) {
                               Yval = Y2array(ic,ibatch);
                               Y_ic  = Yarray(ic,ibatch);
                               abs_err = std::abs(Yval - Y_ic);
                               rel_err = abs_err/(1+std::max( std::abs(Yval),std::abs(Y_ic) ));
                       };
                   };
                   max_abserr = std::max( max_abserr,abs_err);
                   max_relerr = std::max( max_relerr,rel_err);



                   if (idebug >= 1) {
                       T const tol = 1.0/(1000.0 * 1000.0);
                       if ((abs_err > tol ) || (rel_err > tol)) {
                             std::cout  << " idim = " << idim
                                        << " ic = " << ic 
                                        << " Y_ic = " << Y_ic
                                        << " Yval =  " << Yval
                                        << " rel_err =  " << rel_err
                                        << " abs_err = " << abs_err << "\n";
                       };
                     };

                   }; // end for ic


      };



        // -------
        // cleanup
        // -------

        myfree( dAarray_ ); dAarray_ = nullptr;
        myfree( dAparray_ ); dAparray_ = nullptr;

        myfree( dXarray_ ); dXarray_ = nullptr;
        myfree( dYarray_ ); dYarray_ = nullptr;
        myfree( dZarray_ ); dZarray_ = nullptr;
        myfree( dWarray_ ); dWarray_ = nullptr;

        free( Aarray_ ); Aarray_ = nullptr;
        free( Aparray_ ); Aparray_ = nullptr;

        free( Xarray_ ); Xarray_ = nullptr;
        free( Yarray_ ); Yarray_ = nullptr;
        if (use_overlap_in_Y) {
          free( Y2array_ ); Y2array_ = nullptr;
        };

        free( Zarray_ ); Zarray_ = nullptr;
        free( Warray_ ); Warray_ = nullptr;

        return(std::min(max_abserr,max_relerr)); 

}


                      
template<typename T>
int main_func( double const tol) {

        int const idebug = 0;

        int batch_table[] = {1};
        int const size_batch_table = sizeof(batch_table)/sizeof(batch_table[0]);



        int nerrors = 0;
	int const mmax = 2;
	int const nmax = 2;

        for (int idim =1; idim <= 6; idim++) {
        for (int ibatch_table=0; ibatch_table < size_batch_table; ibatch_table++) {
	  int const batchCount = batch_table[ibatch_table];

          int const m1max = (idim >= 1) ? mmax : 1; 
          int const m2max = (idim >= 2) ? mmax : 1; 
          int const m3max = (idim >= 3) ? mmax : 1; 
          int const m4max = (idim >= 4) ? mmax : 1; 
          int const m5max = (idim >= 5) ? mmax : 1; 
          int const m6max = (idim >= 6) ? mmax : 1; 

	  int const n1max = (idim >= 1) ? nmax : 1;
	  int const n2max = (idim >= 2) ? nmax : 1;
	  int const n3max = (idim >= 3) ? nmax : 1;
	  int const n4max = (idim >= 4) ? nmax : 1;
	  int const n5max = (idim >= 5) ? nmax : 1;
	  int const n6max = (idim >= 6) ? nmax : 1;


	  for(int m1=1; m1 <= m1max; m1++)
	  for(int m2=1; m2 <= m2max; m2++)
	  for(int m3=1; m3 <= m3max; m3++)
	  for(int m4=1; m4 <= m4max; m4++)
	  for(int m5=1; m5 <= m5max; m5++)
#ifdef EBUG
	  for(int m6=1; m6 <= m6max; m6++) {
		  int const n1 = m1;
		  int const n2 = m2;
		  int const n3 = m3;
		  int const n4 = m4;
		  int const n5 = m5;
		  int const n6 = m6;
#else
	  for(int m6=1; m6 <= m6max; m6++)  
		  
	  for(int n1=1; n1 <= n1max; n1++)
	  for(int n2=1; n2 <= n2max; n2++)
	  for(int n3=1; n3 <= n3max; n3++)
	  for(int n4=1; n4 <= n4max; n4++)
	  for(int n5=1; n5 <= n5max; n5++)
	  for(int n6=1; n6 <= n6max; n6++) {

#endif
		int const m_array[] = {m1,m2,m3,m4,m5,m6};
	        int const n_array[] = {m1,m2,m3,m4,m5,m6};

                double const max_abserr =  test_kronmult_vbatched<T>( idim, m_array,n_array, batchCount, idebug );
                bool const isok = (max_abserr < tol);
                if (!isok) {
                        nerrors += 1;
                };

                if ((idebug >= 1) || (!isok)) {
                        std::cout << " idim = "  << idim
                                  << " m1,n1 " << m1 << "," << n1
                                  << " m2,n2 " << m2 << "," << n2
                                  << " m3,n3 " << m3 << "," << n3
                                  << " m4,n4 " << m4 << "," << n4
                                  << " m5,n5 " << m5 << "," << n5
                                  << " m6,n6 " << m6 << "," << n6
                                  << " batchCount = " << batchCount
                                  << " max_abserr= " << max_abserr 
				  << "\n";
                };
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
               int const idim = 6;


               for(int n=4; n <= 6; n++) {
		int const m_array[] = {n,n,n,n,n,n};
		int const n_array[] = {n,n,n,n,n,n};
                test_kronmult_vbatched<T>(idim,m_array,n_array, batchCount, idebug, do_check );
               };
        };




  return(0);
}


                     

int main()
{
  double const dtol = 300.0/(1000.0 * 1000.0 *1000.0);
  main_func<double>( dtol );

  double const stol = 10.0/(1000.0 * 1000.0);
  main_func<float>( stol );
}

