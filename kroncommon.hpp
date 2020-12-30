#ifndef KRONCOMMON_HPP
#define KRONCOMMON_HPP 1



#ifdef USE_GPU
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#define GLOBAL_FUNCTION  __global__ 
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#define HOST_FUNCTION __host__

#ifdef __NVCC__
#define WARPSIZE 32
#else
#define WARPSIZE 64
#endif

#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE_FUNCTION
#define HOST_FUNCTION
#endif

#include <cassert>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string.h>

#ifndef USE_GPU

static inline
double atomicAdd(double volatile *p, double dvalue)
{
        double oldvalue = 0;
#ifdef _OPENMP
        #pragma omp atomic capture
#endif
        {
        oldvalue = (*p);
        (*p) += dvalue;
        }
        return(oldvalue);
}

static inline
float atomicAdd( float volatile *p, float dvalue)
{
        float oldvalue = 0;
#ifdef _OPENMP
        #pragma omp atomic capture
#endif
        {
        oldvalue = (*p);
        (*p) += dvalue;
        }
        return(oldvalue);
}


#endif

#ifdef USE_LAMBDA
static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx2f( int const i, 
            int const j, 
            int const ld )
{
         // return( i-1+(j-1)*ld );
	 return( i + j*ld - (1 + ld) );
}
#else

#define indx2f(i,j,ld)  (  (i) + (j) * (ld) - (1 + (ld)) )

#endif


static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx3f( int const i1, 
            int const i2, 
            int const i3,

            int const n1,
            int const n2 )
{
    bool const use_horners_rule = true;
    if (use_horners_rule) {
     return( ((i3-1)*n2 + (i2-1))*n1-1+i1 ); 
    }
    else {
      return(indx2f(i1,i2,n1) + 
	    ((i3)-1)*((n1)*(n2)) );
    };
}


static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx4f( int const i1,
            int const i2,
            int const i3,
            int const i4,

            int const n1,
            int const n2,
            int const n3 )
{
  bool const use_horners_rule = true;
  int ans = 0;
  if (use_horners_rule) {
    ans = ( (((i4-1)*n3 + (i3-1))*n2 + (i2-1))*n1 - 1 + i1 );
    }
  else {
    ans = (indx3f(i1,i2,i3,n1,n2) + 
	  ((i4)-1)*((n1)*(n2)*(n3)) );
  };
  return(ans);
}



static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx5f( int const i1,
            int const i2,
            int const i3,
            int const i4,
            int const i5,

            int const n1,
            int const n2,
            int const n3,
            int const n4 ) 
{
  bool const use_horners_rule = true;
  int ans = 0;
  if (use_horners_rule) {
         ans = ( ((((i5-1)*n4 + (i4-1))*n3 + (i3-1))*n2 + (i2-1))*n1 -1 + i1 );
  }
  else {
	ans = ( indx4f(i1,i2,i3,i4,  n1,n2,n3) + 
		  (i5-1)*(((n1*n2)*n3)*n4) );
  };
  return(ans);

}
        

static inline
HOST_FUNCTION DEVICE_FUNCTION
int indx6f(int const i1,
           int const i2,
           int const i3,
           int const i4,
           int const i5,
           int const i6,

           int const n1,
           int const n2,
           int const n3,
           int const n4,
           int const n5)
{
  bool const use_horners_rule = true;
  int ans = 0;
  if (use_horners_rule) {
        ans = ( (((((i6-1)*n5 + (i5-1))*n4 + (i4-1))*n3 + (i3-1))*n2 + (i2-1))*n1 -1 + i1);
  }
  else {
	ans = ( indx5f(i1,i2,i3,i4,i5, n1,n2,n3,n4) +
		(i6-1)*((((n1*n2)*n3)*n4)*n5) );
  };
  return(ans);
}







#endif
