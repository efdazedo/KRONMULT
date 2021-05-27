#ifndef KRONMULT4_HPP
#define KRONMULT4_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A4)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A4) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void kronmult4(int const n, int const nvec, T const A1_[],
                               T const A2_[], T const A3_[], T const A4_[],
                               T X_[], T Y_[], T W_[], int const lda_in = 0)
// -----------------
// note A1 is n by n
//      X is (n^4 by nvec)
// -----------------
{
  int constexpr ndim = 4;
  kronmultx<T, ndim>(n, nvec, A1_, A2_, A3_, A4_, nullptr, nullptr, X_, Y_, W_,
                     lda_in);
}

#endif
