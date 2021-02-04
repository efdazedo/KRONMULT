#ifndef KRONMULT_VBATCHED_HPP
#define KRONMULT_VBATCHED_HPP 1

#include "kroncommon.hpp"

#include "kronmultv1.hpp"
#include "kronmultv2.hpp"
#include "kronmultv3.hpp"
#include "kronmultv4.hpp"
#include "kronmultv5.hpp"
#include "kronmultv6.hpp"



// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A6(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T,int ndim>
DEVICE_FUNCTION
void kronmult_vbatched(
		       int const m1_in, int const n1_in, 
		       int const m2_in, int const n2_in, 
		       int const m3_in, int const n3_in, 
		       int const m4_in, int const n4_in, 
		       int const m5_in, int const n5_in, 
		       int const m6_in, int const n6_in, 
		       T const * const Aarray_[],
                       T* pX_[],
                       T* pY_[],
                       T* W_,
		       size_t const Wcapacity_bytes,
                       int const batchCount_in
		       )
//
// conceptual shape of Aarray is  (ndim,batchCount)
// A1 = Aarray(1,ibatch) is m(1) by n(1)
// A2 = Aarray(2,ibatch) is m(2) by n(2)
// ...
//
// pX_[] is array of pointers to X[], size is n(1)*...*n(6)
// pY_[] is array of pointers to Y[], size is m(1)*...*m(6)
// pW_[] is array of pointers to Z[], each of size 
//    size of W  max( n(2)*..*n(6)*m(1),
//                    n(3)*..*n(6)*m(1)*m(2)
//                    n(4)*..*n(6)*m(1)*m(2)*m(3)
//                    n(5)*..*n(6)*m(1)*m(2)*m(3)*m(4)
//                    n(6)*m(1)*m(2)*m(3)*m(4)*m(5)
//                   )
//   also need sufficient storage to keep a copy of X(i) of size n(1)*..*n(6)
//   thus
//   need temporary storage for pW_[] to be { 2*max(sizeW , sizeX) } * subbatchCount
//
// Y is the output
// X is the input 
// W is workspace
//
//
{
	int const idebug = 0;

	int const m1 = (ndim >= 1) ? m1_in : 1; int const n1 = (ndim >= 1) ? n1_in : 1;
	int const m2 = (ndim >= 2) ? m2_in : 1; int const n2 = (ndim >= 2) ? n2_in : 1;
	int const m3 = (ndim >= 3) ? m3_in : 1; int const n3 = (ndim >= 3) ? n3_in : 1;
	int const m4 = (ndim >= 4) ? m4_in : 1; int const n4 = (ndim >= 4) ? n4_in : 1;
	int const m5 = (ndim >= 5) ? m5_in : 1; int const n5 = (ndim >= 5) ? n5_in : 1;
	int const m6 = (ndim >= 6) ? m6_in : 1; int const n6 = (ndim >= 6) ? n6_in : 1;
#ifdef USE_GPU
        // -------------------------------------------
        // note 1-based matlab convention for indexing
        // -------------------------------------------
        int const iz_start = blockIdx.x + 1;
        int const iz_size =  gridDim.x;
        assert( gridDim.y == 1);
        assert( gridDim.z == 1);

        int const ix_start = threadIdx.x + 1;
        int const ix_size = blockDim.x;
        assert( blockDim.y == 1);
        assert( blockDim.z == 1);
#else
        int const iz_start = 1;
        int const iz_size = 1;

        int const ix_start = 1;
        int const ix_size = 1;
#endif

	auto min = [] (int const x, int const y) -> int {
		return( (x < y) ? x : y );
	};

	auto max = [] (int const x, int const y) -> int {
		return( (x > y) ? x : y );
	};



        auto Aarray = [=] (int const i1,
                           int const i2
                           ) -> T const * const  {
                return( Aarray_[ indx2f(i1,i2,ndim ) ] );
        };





	auto copyT = [=](T       * const dest,
                         T const * const src,     
			  int const nitems) {
                assert( nitems >= 0 );
                assert( dest != nullptr );
                assert( src != nullptr );

                SYNCTHREADS;
		for(int ix=ix_start; ix <= nitems; ix += ix_size)  {
			dest[ix-1] = src[ix-1];
		};
                SYNCTHREADS;
	};






     // -------------------
     // size of W is max of 
     // m(1)*n(2)..n(6)
     // m(1)*m(2)*n(3)..n(6)
     // ...
     // m(1)*m(2)..m(5)*n(6)
     // -------------------
	int const n56    = n5 * n6;
	int const n456   = n4 * n56;
	int const n3456  = n3 * n456;
	int const n23456 = n2 * n3456;
	int sizeX = n1 * n23456;

	int m12 = m1 * m2;
	int m123 = m12 * m3;
	int m1234 = m123 * m4;
	int m12345 = m1234 * m5;

	int m1n6 = m1 * n23456;
	int m2n6 = m12 * n3456;
	int m3n6 = m123 * n456;
	int m4n6 = m1234 * n56;
	int m5n6 = m12345 * n6;

        int const sizeW = max(  max( max(m1n6,m2n6), max(m3n6,m4n6)), m5n6);
		     

	int const sizeXW = max( sizeX, sizeW );
	int const subbatchCount =  max(1,min( batchCount_in, (Wcapacity_bytes/(2*sizeXW*sizeof(T)) )));
	if (idebug >= 1) {
		printf("sizeX=%d, sizeW=%d, subbatchCount=%d Wcapacity_bytes=%ld\n",
		        sizeX,    sizeW,    subbatchCount,   Wcapacity_bytes );
	};
	assert( subbatchCount >= 1 );


        for(int ibatch_start = 1; ibatch_start <= batchCount_in; ibatch_start += subbatchCount) {
		int const ibatch_end = min( batchCount_in , ibatch_start + subbatchCount-1);
		int const batchCount = ibatch_end - ibatch_start + 1;

#if defined(_OPENMP) && !defined(USE_GPU)
        #pragma omp parallel for
#endif
        for(int iz=iz_start; iz <= batchCount; iz += iz_size) {
		int ibatch = (ibatch_start-1) + iz;
                T* const Xp_in =  pX_[ (ibatch-1) ];
                

		// -----------------
		// storage of W_ contains
		// (X(1), W(1))  so occupy sizeXW + sizeXW
		// (X(2), W(2))
		// ...
		// (X(subbatchCount), W(subbatchCount))
		// -----------------
		T* const Xp = &(W_[ (iz-1)*(sizeXW+sizeXW) ]);
                T* const Wp = Xp + sizeXW;


		{
                T const * const src = Xp_in;
		T       * const dest = Xp;
		int nitems =  sizeX;
		copyT( dest, src, nitems );
		}

                T* const Yp =  pY_[ (ibatch-1) ];

                T const * const A1 = (ndim >= 1) ? (Aarray(1,ibatch)) : nullptr;
                T const * const A2 = (ndim >= 2) ? (Aarray(2,ibatch)) : nullptr;
                T const * const A3 = (ndim >= 3) ? (Aarray(3,ibatch)) : nullptr;
                T const * const A4 = (ndim >= 4) ? (Aarray(4,ibatch)) : nullptr;
                T const * const A5 = (ndim >= 5) ? (Aarray(5,ibatch)) : nullptr;
                T const * const A6 = (ndim >= 6) ? (Aarray(6,ibatch)) : nullptr;



		int const ld1 = m1;
		int const ld2 = m2;
		int const ld3 = m3;
		int const ld4 = m4;
		int const ld5 = m5;
		int const ld6 = m6;

                int const nvec = 1;
		switch(ndim)
                {
		case 1: {kronmultv1( m1,n1,A1,ld1,                
				     nvec, Xp, Yp, Wp ); break;}
		case 2: {kronmultv2( m1,n1,A1,ld1,
				     m2,n2,A2,ld2,                
				     nvec, Xp, Yp, Wp ); break;}
		case 3: {kronmultv3( m1,n1,A1,ld1,
				     m2,n2,A2,ld2,                
				     m3,n3,A3,ld3,                
				     nvec, Xp, Yp, Wp ); break;}
		case 4: {kronmultv4( m1,n1,A1,ld1,
				     m2,n2,A2,ld2,                
				     m3,n3,A3,ld3,                
				     m4,n4,A4,ld4,                
				     nvec, Xp, Yp, Wp ); break;}
		case 5: {kronmultv5( m1,n1,A1,ld1,
				     m2,n2,A2,ld2,                
				     m3,n3,A3,ld3,                
				     m4,n4,A4,ld4,                
				     m5,n5,A5,ld5,                
				     nvec, Xp, Yp, Wp ); break;}
		case 6: {kronmultv6( m1,n1,A1,ld1,
				     m2,n2,A2,ld2,                
				     m3,n3,A3,ld3,                
				     m4,n4,A4,ld4,                
				     m5,n5,A5,ld5,                
				     m6,n6,A6,ld6,                
				     nvec, Xp, Yp, Wp ); break;}
                default: { assert( false ); }
		};
        }; // for iz
	}; // for ibatch_start

}



#endif
