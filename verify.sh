export OMP_NUM_THREADS=8
export OMP_STACKSIZE=2G
make clean;
make ;
./test_kgemm_nn_batched
./test_kgemm_nt_batched
./test_kronmult6_batched
./test_kronmult6_pbatched
./test_kronmult6_xbatched
