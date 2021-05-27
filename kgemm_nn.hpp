#ifndef KGEMM_NN_HPP
#define KGEMM_NN_HPP 1

#include "kroncommon.hpp"

//  -----------------------
//  NotransA and TransB case
//  C = alpha*A*(B) + beta *C
//  -----------------------
//

template<typename T, typename Tc>
DEVICE_FUNCTION void
kgemm_nn2(int const mm, int const nn, int const kk, T const alpha_in,
          T const *const A_, int const ldA, T const *const B_, int const ldB,
          T const beta_in, T *C_, int const ldC)
{
#ifdef USE_LAMBDA
  auto min = [](int const x, int const y) { return ((x < y) ? x : y); };
  auto max = [](int const x, int const y) { return ((x > y) ? x : y); };
#else

#ifndef min
#define min(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef max
#define max(x, y) (((x) > (y)) ? (x) : (y))
#endif

#endif

  Tc const alpha = alpha_in;
  Tc const beta  = beta_in;

  int constexpr nb = 2 * 32;
#ifdef USE_GPU
  // ---------------------------
  // use matlab 1 based indexing
  // ---------------------------

  int constexpr warpsize = WARPSIZE;
  int const nthreads     = blockDim.x;

  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert((nthreads % warpsize) == 0);

  // -----------------------------------------
  // -----------------------------------------

  int const ij_start = threadIdx.x + 1;
  int const ij_size  = nthreads;
#else

  int const ij_start = 1;
  int const ij_size  = 1;
#endif

  //  ------------------------------------
  //  commonly  nn is large, but kk, nn are small
  //
  //  consider increasing nb for more effective
  //  use of shared cache
  //
  //  ------------------------------------

#ifdef USE_LAMBDA
  auto A = [=](int const ia, int const ja) -> T const & {
    return (A_[indx2f(ia, ja, ldA)]);
  };

  auto B = [=](int const ib, int const jb) -> T const & {
    return (B_[indx2f(ib, jb, ldB)]);
  };

  auto C = [=](int const ic, int const jc) -> T & {
    return (C_[indx2f(ic, jc, ldC)]);
  };

#else

#define A(ia, ja) A_[indx2f(ia, ja, ldA)]
#define B(ib, jb) B_[indx2f(ib, jb, ldB)]
#define C(ic, jc) C_[indx2f(ic, jc, ldC)]

#endif

  for (int istart = 1; istart <= mm; istart += nb)
  {
    int const iend  = min(mm, istart + nb - 1);
    int const isize = iend - istart + 1;

    for (int jstart = 1; jstart <= nn; jstart += nb)
    {
      int const jend  = min(nn, jstart + nb - 1);
      int const jsize = jend - jstart + 1;

      SYNCTHREADS;

      // ---------------------------
      // perform matrix calculations
      // ---------------------------

      // Compute (1) C = A * X, A is k by k, X is k by n, n >> k
      //         C(i,j) = sum( A(i,k) * X(k,j), over k )
      //
      //         (2) C = X * A, A is k by k, X is n by k, n >> k
      //         C(i,j) = sum( X(i,k) * A(k,j), over k )
      bool use_i_faster = (isize >= jsize);
      for (int ij0 = ij_start - 1; ij0 < (isize * jsize); ij0 += ij_size)
      {
        int i, j;
        if (use_i_faster)
        {
          // -------------------------
          // ij0 = (i-1) + (j-1)*isize
          // -------------------------
          j = (ij0 / isize) + 1;
          i = (ij0 - (j - 1) * isize) + 1;
          assert(ij0 == ((i - 1) + (j - 1) * isize));
        }
        else
        {
          // -------------------------
          // ij0 = (j-1) + (i-1)*jsize
          // -------------------------
          i = (ij0 / jsize) + 1;
          j = (ij0 - (i - 1) * jsize) + 1;
          assert(ij0 == ((j - 1) + (i - 1) * jsize));
        };

        int const ia = (istart - 1) + i;
        int const jb = (jstart - 1) + j;

        auto const inc_A           = ldA;
        auto const inc_B           = 1;
        Tc cij                     = 0;
        bool constexpr use_pointer = true;
        if (use_pointer)
        {
          int const k = 1;
          T const *Ap = &(A(ia, k));
          T const *Bp = &(B(k, jb));

#define case_code(kk)            \
  {                              \
    for (int k = 0; k < kk; k++) \
    {                            \
      Tc const aik = (*Ap);      \
      Tc const bkj = (*Bp);      \
      cij += aik * bkj;          \
      Ap += inc_A;               \
      Bp += inc_B;               \
    };                           \
    break;                       \
  }

          switch (kk)
          {
          case 1:
            case_code(1) case 2 : case_code(2) case 3 : case_code(3) case 4
                : case_code(4) case 5 : case_code(5) case 6
                : case_code(6) case 7 : case_code(7) case 8
                : case_code(8) default : case_code(kk);
          };
        }
        else
        {
          for (int k = 1; k <= kk; k++)
          {
            Tc const aik = A(ia, k);
            Tc const bkj = B(k, jb);
            cij += aik * bkj;
          };
        };
        // ------------------
        // store results to C
        // ------------------
        int const ic = ia;
        int const jc = jb;
        T alpha_cij  = alpha * cij;
        if (beta == 1)
        {
          atomicAdd(&(C(ic, jc)), alpha_cij);
        }
        else if (beta == 0)
        {
          C(ic, jc) = alpha_cij;
        }
        else
        {
          C(ic, jc) = beta * C(ic, jc) + alpha_cij;
        };
      };

    }; // end istart
  };   // end jstart

  SYNCTHREADS;
}

template<typename T>
DEVICE_FUNCTION void
kgemm_nn(int const mm, int const nn, int const kk, T const alpha,
         T const *const A_, int const ldA, T const *const B_, int const ldB,
         T const beta, T *C_, int const ldC)
{
  kgemm_nn2<T, T>(mm, nn, kk, alpha, A_, ldA, B_, ldB, beta, C_, ldC);
}

template<>
DEVICE_FUNCTION void
kgemm_nn(int const mm, int const nn, int const kk, float const alpha,
         float const *const A_, int const ldA, float const *const B_,
         int const ldB, float const beta, float *C_, int const ldC)
{
  kgemm_nn2<float, double>(mm, nn, kk, alpha, A_, ldA, B_, ldB, beta, C_, ldC);
}

#undef min
#undef max
#undef A
#undef B
#undef C
#undef case_code

#endif
