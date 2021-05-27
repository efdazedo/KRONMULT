make clean; make -j
leaks -atExit -- ./test_kronmult6_xbatched >& test_kronmult6_xbatched.out
leaks -atExit -- ./test_kronmult6_pbatched >& test_kronmult6_pbatched.out
leaks -atExit -- ./test_kronmult6_vbatched >& test_kronmult6_vbatched.out
leaks -atExit -- ./test_kronmult6_batched >& test_kronmult6_batched.out
leaks -atExit -- ./test_kgemm_nn_batched >& test_kgemm_nn_batched.out
leaks -atExit -- ./test_kgemm_nt_batched >& test_kgemm_nt_batched.out
leaks -atExit -- ./test_kgemm_tt_batched >& test_kgemm_tt_batched.out


