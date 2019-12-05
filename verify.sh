make clean
make
export OMP_NUM_THREADS=8
./test_kgemm_nn_batched
./test_kgemm_nt_batched
./test_kronmult6_batched
