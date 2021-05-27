#ifndef KRONMULTV1_HPP
#define KRONMULTV1_HPP 1

#include "kroncommon.hpp"

#include "kgemm_nn.hpp"

//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1)*X as
//  which is just matrix multiply  Y = A1 * X
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION void
kronmultv1(int const m1, int const n1, T const A1_[], int const ld1,
           int const nvec, T X_[], T Y_[], T W_[])
// -----------------
// note A1 is m1 by n1
//      X is (n1 by nvec)
//      Y(1:m1,1:nvec) = A(1:m1,1:n1) * X(1:n1, 1:nvec)
// -----------------
{
  // used to suppress warnings in unused variables
  auto const ignore = [](T *ignored) { (void)ignored; };
  ignore(W_);

  int const mm = m1;
  int const nn = nvec;
  int const kk = n1;

  int const ldAp = ld1;
  int const ldBp = kk;
  int const ldCp = mm;

  T const *const Ap = &(A1_[0]);
  T const *const Bp = &(X_[0]);
  T *const Cp       = &(Y_[0]);

  T const alpha = 1;
  T const beta  = 1;

  kgemm_nn(mm, nn, kk, alpha, Ap, ldAp, Bp, ldBp, beta, Cp, ldCp);
}

#endif
