Code to perform 6-dimensional batched kronecker product on  GPU and CPU (using OpenMP).

Y(:,k) += kron(A1(:,:,k), ..., A6(:,:,k) ) * X(:,k),   where k=1:batchCount

The code assumes  each matrix A1, ..., A6 are all square and same shape of n by n.

Note there can be overlap in the output vectors Y(:,k). Atomic add updates will be performed.

Each vector X(:,k) or Y(:,k) is conceptually  of shape n^6 by 1 but during the computation
can be reshaped as   n^5 by n or   n^4 by n^2 or n^3 by n^3 or n^2 by n^4 or n by n^5.

Note that  
Y = kron(A1,...,A6) * X
can be evaluated recursively as
step 1: W = reshape( X, [n^5,n]) * transpose(A1)
step 2: Y = kron( A2, ..., A6) * W,   which can be viewed as "tail
recursion" for computing 5-dimensional Kronecker products with multiple
vectors. To avoid data rearrangement, the lower dimensional Kronecker
product for multiple vector is performed in a loop with 1 vector at
a time.

At the lowest level of recursion
Y = kron(A1) * X  is simply   implemented as matrix multiply 
Y = A1 * X  

-------------

Another algorithm is based on "Algorithm 993: Efficient Computation with Kronecker Products"
by Paul L. Fackler  in ACM Transactions on Mathematical Software, Volume 45, Issue 2
(https://dl.acm.org/doi/10.1145/3291041)
by performing ndim  matrix multiplications  for a ndim-dimensional Kronecker product.

The forward variant performs the operations

for i=1:ndim
  % note:  perform atomic update at final iteration
  Yout = A(i) * Xin';
  swap( Xin, Yout);
end;

The backward variant performs the operations

for j=1:ndim,
  i = (ndim-j)+1;
  Yout = Xin' * A(i)';
  % note: perform atomic update at final iteration
  swap( Xin, Yout );
end;


If all A1, ..., A6 are all square and same size, then the forward variant
and backward variant will perform the same amount of work. However,
if each A1, ..., A6 can be rectangular and different shapes, then one
variant may perform less work.


---------------

Implementation Details:


This implementation evaluates each batch entry Kronecker product  in a separate
thread block on GPU, or over an OpenMP parallel loop on CPU. Instead
of building a long batch list to call batched GEMM, the matrix-matrix
multiplication is evaluated as calls to device functions

kgemm_nn() to evaluate  C = alpha * A * B + beta * C
or
kgemm_nt() to evaluate  C = alpha * A * transpose(B) + beta * C
or
kgemm_tt() to evaluate  C = alpha * transpose(A) * transpose(B) + beta * C

For the special case of (alpha == 1 and beta == 1), atomicAdd update is used.

Essentially the same code is used on GPU and CPU by making minor
adjustments in the specification of "for" loops but keeping the loop body
unchanged. This simplifies debugging but may yield non-optimal performance
compared to using separate versions for GPU and CPU.  The GPU code is
implemented using Heterogeneous-Computer Interface for Portability (HIP)
to  compile using cuda on Nvidia GPU such as V100,  and to compile using
ROCm on AMD GPU such as MI50.


Note that the GEMM operations will be performed on very slender
rectangular matrices.  Therefore, the computations will likely not be
dominated by floating point operations but by cache reuse and data
movement.  This is especially the case when the sizes of matrices
A1,...,A6 are small.



-----------------

To compile the code for CPU
(1) mkdir build && cd build
(2) 
    cmake -DUSE_ALG993 ../ # to use algorithm 993
or
    cmake -UUSE_ALG993 ../ # to use recursive algorithm
(3) make

To compile the code for Nvidia GPU
(1) mkdir build && cd build
(2) 
cmake ../ -DUSE_GPU=1 -DUSE_ALG993 # to use algorithm 993
or
cmake ../ -DUSE_GPU=1 -UUSE_ALG993 # to use recursive algorithm
(3) make

To run the tester for kgemm_nn_batched, perform
./test_kgemm_nn_batched

To run the tester for kgemm_nt_batched, perform
./test_kgemm_nt_batched

To run the tester for kronmult6_batched, perform
./test_kronmult6_batched



