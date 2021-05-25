export OMP_NUM_THREADS=8
export OMP_STACKSIZE=2G
make clean;
make -j  ;
echo " === test_kgemm_nn_batched ==== "
./test_kgemm_nn_batched
echo " === test_kgemm_nt_batched ==== "
./test_kgemm_nt_batched
echo " === test_kgemm_tt_batched ==== "
./test_kgemm_tt_batched

echo " === test_kronmult6_batched ==== "
./test_kronmult6_batched


echo " === test_kronmult6_pbatched ==== "
./test_kronmult6_pbatched


echo " === test_kronmult6_xbatched ==== "
./test_kronmult6_xbatched

echo " === test_kronmult6_vbatched ==== "
./test_kronmult6_vbatched
