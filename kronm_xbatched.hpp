#ifndef KRONM_XBATCHED_HPP
#define KRONM_XBATCHED_HPP 1

#include "kroncommon.hpp"
#include "kronm.hpp"

// --------------------------------------------------------------------
// Performs  Y(:,k) += kron(A1(k),...,A6(k)) * X(:,k), k=1:batchCount
// Note  result in Y but X and W may be modified as temporary work space
// --------------------------------------------------------------------
template<typename T, int ndim>
DEVICE_FUNCTION void
kronm_xbatched(int const n, T const *const Aarray_[], int const lda, T *pX_[],
               T *pY_[], T *pW_[], int const batchCount_in,
               int subbatchCount = 0)
//
// conceptual shape of Aarray is  (ndim,batchCount)
//
// pX_[] is array of pointers to X[], each of size n^ndim
// pY_[] is array of pointers to Y[], each of size n^ndim
// pW_[] is array of pointers to Z[], each of size n^ndim
//
// Y is the output
// X is the input
// W is workspace
//
//
{
#ifdef USE_GPU
  // -------------------------------------------
  // note 1-based matlab convention for indexing
  // -------------------------------------------
  int const iz_start = blockIdx.x + 1;
  int const iz_size  = gridDim.x;
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);

  int const ix_start = threadIdx.x + 1;
  int const ix_size  = blockDim.x;
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
#else
  int const iz_start = 1;
  int const iz_size  = 1;

  int const ix_start = 1;
  int const ix_size  = 1;
#endif

  auto min = [](int const x, int const y) -> int { return ((x < y) ? x : y); };

  auto max = [](int const x, int const y) -> int { return ((x > y) ? x : y); };

  if (subbatchCount == 0)
  {
    subbatchCount = max(1, batchCount_in / 2);
  };

  assert(subbatchCount >= 1);

  auto Aarray = [=](int const i1, int const i2) -> T const *const {
    return (Aarray_[indx2f(i1, i2, ndim)]);
  };

  auto copyT = [=](T *const dest, T const *const src, int const nitems) {
    assert(nitems >= 0);
    assert(dest != nullptr);
    assert(src != nullptr);

    SYNCTHREADS;
    for (int ix = ix_start; ix <= nitems; ix += ix_size)
    {
      dest[ix - 1] = src[ix - 1];
    };
    SYNCTHREADS;
  };

  auto pow = [=](int const x, int const d) -> int {
    // compute x^d
    assert(d >= 0);
    int result = 1;
    for (int i = 0; i < d; i++)
    {
      result *= x;
    };
    return (result);
  };

  int const n_to_ndim = pow(n, ndim);
  for (int ibatch_start = 1; ibatch_start <= batchCount_in;
       ibatch_start += subbatchCount)
  {
    int const ibatch_end = min(batchCount_in, ibatch_start + subbatchCount - 1);
    int const batchCount = ibatch_end - ibatch_start + 1;

#if defined(_OPENMP) && !defined(USE_GPU)
#pragma omp parallel for
#endif
    for (int iz = iz_start; iz <= batchCount; iz += iz_size)
    {
      int ibatch     = (ibatch_start - 1) + iz;
      T *const Xp_in = pX_[(ibatch - 1)];

      T *const Xp = (batchCount_in > 1) ? pW_[2 * (iz - 1)] : Xp_in;
      T *const Wp =
          (batchCount_in > 1) ? pW_[2 * (iz - 1) + 1] : pW_[(ibatch - 1)];

      {
        T const *const src = Xp_in;
        T *const dest      = Xp;
        int nitems         = n_to_ndim;
        copyT(dest, src, nitems);
      }

      T *const Yp = pY_[(ibatch - 1)];

      T const *const A1 = (ndim >= 1) ? (Aarray(1, ibatch)) : nullptr;
      T const *const A2 = (ndim >= 2) ? (Aarray(2, ibatch)) : nullptr;
      T const *const A3 = (ndim >= 3) ? (Aarray(3, ibatch)) : nullptr;
      T const *const A4 = (ndim >= 4) ? (Aarray(4, ibatch)) : nullptr;
      T const *const A5 = (ndim >= 5) ? (Aarray(5, ibatch)) : nullptr;
      T const *const A6 = (ndim >= 6) ? (Aarray(6, ibatch)) : nullptr;
      int const nvec    = 1;

      T const *const A_array[] = {A1, A2, A3, A4, A5, A6};
      int const m              = n;
      int m_array[]            = {m, m, m, m, m, m};
      int n_array[]            = {n, n, n, n, n, n};
      kronm<T>(ndim, m_array, n_array, &(A_array[0]), nvec, Xp, Yp, Wp);
    }; // for iz
  };   // for ibatch_start
}

#endif
