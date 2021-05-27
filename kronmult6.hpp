#ifndef KRONMULT6_HPP
#define KRONMULT6_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nt.hpp"
#include "kronmultx.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X as
//  W(:,k) = X(:,k) * transpose(A1), k=1:nvec
//  Y = kron(A2,..,A6) * W
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void
kronmult6(int const n, int const nvec, T const A1_[], T const A2_[],
          T const A3_[], T const A4_[], T const A5_[], T const A6_[], T X_[],
          T Y_[], T W_[], int const lda_in = 0)
// -----------------
// note A1 is n by n
//      X is (n^6 by nvec)
// -----------------
{
  int constexpr ndim = 6;
  kronmultx<T, ndim>(n, nvec, A1_, A2_, A3_, A4_, A5_, A6_, X_, Y_, W_, lda_in);
}

#endif
