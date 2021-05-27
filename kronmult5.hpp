#ifndef KRONMULT5_HPP
#define KRONMULT5_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A5)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A5) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void
kronmult5(int const n, int const nvec, T const A1_[], T const A2_[],
          T const A3_[], T const A4_[], T const A5_[], T X_[], T Y_[], T W_[],
          int const lda_in = 0)
// -----------------
// note A1 is n by n
//      X is (n^5 by nvec)
// -----------------
{
  int constexpr ndim = 5;
  kronmultx<T, ndim>(n, nvec, A1_, A2_, A3_, A4_, A5_, nullptr, X_, Y_, W_,
                     lda_in);
}

#endif
