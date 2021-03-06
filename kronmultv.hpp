#ifndef KRONMULTV_HPP
#define KRONMULTV_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nn.hpp"
#include "kgemm_nt.hpp"

#include "kronm.hpp"
#include "kronmultv_recur.hpp"

template<typename T, int ndim = 1>
DEVICE_FUNCTION void kronmultv(

    int const m1, int const n1, T const A1_[], int const ld1, int const m2,
    int const n2, T const A2_[], int const ld2, int const m3, int const n3,
    T const A3_[], int const ld3, int const m4, int const n4, T const A4_[],
    int const ld4, int const m5, int const n5, T const A5_[], int const ld5,
    int const m6, int const n6, T const A6_[], int const ld6, int const nvec,
    T X_[], T Y_[], T W_[])
{
#ifdef USE_ALG993
  bool constexpr use_alg993 = true;
#else
  bool constexpr use_alg993 = false;
#endif

  if (use_alg993)
  {
    // -----------------------
    // use kronm algorithm 993
    // -----------------------
    int const m_array[]      = {m1, m2, m3, m4, m5, m6};
    int const n_array[]      = {n1, n2, n3, n4, n5, n6};
    int const ld_array[]     = {ld1, ld2, ld3, ld4, ld5, ld6};
    T const *const A_array[] = {A1_, A2_, A3_, A4_, A5_, A6_};

    kronm<T>(ndim, m_array, n_array, A_array, ld_array, nvec, X_, Y_, W_);
  }
  else
  {
    kronmultv_recur<T, ndim>(m1, n1, A1_, ld1, m2, n2, A2_, ld2, m3, n3, A3_,
                             ld3, m4, n4, A4_, ld4, m5, n5, A5_, ld5, m6, n6,
                             A6_, ld6, nvec, X_, Y_, W_);
  };
}
#endif
