#ifndef KRONMULT2_HPP
#define KRONMULT2_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A2)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A2) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void
kronmult2(int const n, int const nvec, T const A1_[], T const A2_[], T X_[],
          T Y_[], T W_[], int const lda_in = 0)
// -----------------
// note A1 is n by n
//      X is (n^2 by nvec)
// -----------------
{
  int constexpr ndim = 2;
  kronmultx<T, ndim>(n, nvec, A1_, A2_, nullptr, nullptr, nullptr, nullptr, X_,
                     Y_, W_, lda_in);
}

#endif
