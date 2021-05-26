
#ifndef KRONM_HPP
#define  KRONM_HPP 1

#include "kroncommon.hpp"
#include "kronm_forward.hpp"
#include "kronm_backward.hpp"





//  -------------------------------------------
//  device function to evaluate
//  Y = kron(A1,...,A6)*X 
//
//  using algorithm 993
//  -------------------------------------------
template<typename T>
DEVICE_FUNCTION
void kronm( 
        int const ndim,
        int const m_array[],
        int const n_array[],
        T const * const A_array[],
	int const ld_array[],
        int const nvec,
        T* X_,
        T* Y_,
        T* W_ 
        )
                
{
    bool constexpr use_forward = true;
    int constexpr idebug = 0;
    if (idebug >= 1) {
      printf("kronm:ndim %d, nvec %d\n", ndim,nvec);
    };

    if (use_forward) {
        kronm_forward<T>(ndim, m_array, n_array, A_array, ld_array,
                         nvec, X_, Y_, W_ );
    }
    else {
        kronm_backward<T>(ndim, m_array, n_array, A_array, ld_array,
                         nvec, X_, Y_, W_ );
    };
}
#endif