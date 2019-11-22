#ifndef KRONCOMMON_HPP
#define KRONCOMMON_HPP 1




#include <cassert>

#ifndef MOD
#define MOD(i,n)  ((i) % (n))
#endif

#ifndef ABS
#define ABS(x) ( ((x) >= 0) ? (x) : (-(x)) )
#endif

#ifndef MAX
#define MAX(x,y)  ( ((x) > (y)) ? (x) : (y) )
#endif

#ifndef MIN
#define MIN(x,y)  (  ((x) < (y)) ? (x) : (y) )
#endif

#ifndef indx2f
#define indx2f(i,j,ld) (((i)-1) + ((j)-1)*(ld))
#endif

#ifndef indx3f
#define indx3f(i1,i2,i3,n1,n2) (indx2f(i1,i2,n1) + ((i3)-1)*((n1)*(n2)) )
#endif

#ifndef indx4f
#define indx4f(i1,i2,i3,i4,n1,n2,n3) (indx3f(i1,i2,i3,n1,n2) + ((i4)-1)*((n1)*(n2)*(n3)) )
#endif

#ifndef indx5f
#define indx5f(i1,i2,i3,i4,i5,n1,n2,n3,n4) (indx4f(i1,i2,i3,i4,n1,n2,n3) + ((i5)-1)*((n1)*(n2)*(n3)*(n4)) )
#endif

#ifndef indx6f
#define indx6f(i1,i2,i3,i4,i5,i6,   n1,n2,n3,n4,n5) (indx5f(i1,i2,i3,i4,i5,n1,n2,n3,n4) + ((i6)-1)*((n1)*(n2)*(n3)*(n4)*(n5)) )
#endif

#ifdef USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#define GLOBAL_FUNCTION  __global__ 
#define SYNCTHREADS __syncthreads()
#define SHARED_MEMORY __shared__
#define DEVICE_FUNCTION __device__
#else
#define GLOBAL_FUNCTION
#define SYNCTHREADS 
#define SHARED_MEMORY 
#define DEVICE_FUNCTION
#endif




#endif
