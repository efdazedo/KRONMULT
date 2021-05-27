
#ifndef KRONM_HPP
#define KRONM_HPP 1

#include "kroncommon.hpp"
#include "kronm_backward.hpp"
#include "kronm_forward.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X
//
//  using algorithm 993
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void
kronm(int const ndim, int const m_array[], int const n_array[],
      T const *const A_array[], int const ld_array[], int const nvec, T *X_,
      T *Y_, T *W_)

{
  int const idebug = 0;

  auto prod = [=](int ndim, int const n_array[]) -> int {
    int iprod = 1;
    for (int i = 0; i < ndim; i++)
    {
      iprod *= n_array[i];
    };
    return (iprod);
  };

  auto flops_forward = [=](int ndim, int const m_array[],
                           int const n_array[]) -> double {
    int Xsize          = prod(ndim, n_array);
    double total_flops = 0;
    for (int i = 0; i < ndim; i++)
    {
      // -----------------
      // Yout = Amat * Xin'
      // Amat is m by k, Xin is n by k
      // -----------------

      int const mm       = m_array[i];
      int const kk       = n_array[i];
      int const nn       = Xsize / kk;
      double const flops = (2.0 * mm) * nn * kk;
      total_flops += flops;
      Xsize = mm * nn;
    };
    return (total_flops);
  };

  auto flops_backward = [=](int ndim, int const m_array[],
                            int const n_array[]) -> double {
    int Xsize          = prod(ndim, n_array);
    double total_flops = 0;
    for (int i = (ndim - 1); i >= 0; i--)
    {
      // -----------------
      // Yout = Xin' * Amat'
      // Amat is n by k, Xin is k by m
      // -----------------
      int const nn       = m_array[i];
      int const kk       = n_array[i];
      int const mm       = Xsize / kk;
      double const flops = (2.0 * mm) * nn * kk;
      total_flops += flops;
      Xsize = mm * nn;
    };
    return (total_flops);
  };

  double const dflops_forward  = flops_forward(ndim, m_array, n_array);
  double const dflops_backward = flops_backward(ndim, m_array, n_array);
#ifdef USE_GPU
  // ------------------------------------------------------
  // prefer forward algorithm on GPU for unit stride in "j"
  // ------------------------------------------------------
  bool const use_forward = (dflops_forward <= dflops_backward);
#else
  // ------------------------------------------------------
  // prefer backward algorithm on CPU for unit stride in "k"
  // ------------------------------------------------------
  bool const use_forward = (dflops_forward < dflops_backward);
#endif

  if (idebug >= 1)
  {
    printf("kronm:ndim %d, nvec %d\n", ndim, nvec);
    printf("dflops_forward=%lf, dflops_backward=%lf\n", dflops_forward,
           dflops_backward);
  };

  if (use_forward)
  {
    kronm_forward<T>(ndim, m_array, n_array, A_array, ld_array, nvec, X_, Y_,
                     W_);
  }
  else
  {
    kronm_backward<T>(ndim, m_array, n_array, A_array, ld_array, nvec, X_, Y_,
                      W_);
  };
}
#endif
