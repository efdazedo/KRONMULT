#include "kroncommon.hpp"
#include "kgemm_tt.hpp"

DEVICE_FUNCTION
void kgemm_tt( int const mm, int const nn, int const kk,
          double const alpha,
          double const * const A_, int const ldA,
          double const * const B_, int const ldB,
          double const beta,
          double * const C_, int const ldC )
{


  kgemm_tt<double>( mm,nn,kk,
                   alpha,
                   A_, ldA,
                   B_, ldB,
                   beta,
                   C_, ldC );
}
