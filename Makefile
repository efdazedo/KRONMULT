CXXFLAGS = -DUSE_GPU  -g
NVCC=nvcc -x cu 

all: test_kgemm_nn_batched test_kgemm_nt_batched

test_kgemm_nn_batched: test_kgemm_nn_batched.cpp kgemm_nn_batched.hpp kgemm_nn.hpp
	$(NVCC) $(CXXFLAGS) -o test_kgemm_nn_batched test_kgemm_nn_batched.cpp -lcuda

test_kgemm_nt_batched: test_kgemm_nt_batched.cpp kgemm_nt_batched.hpp kgemm_nt.hpp
	$(NVCC) $(CXXFLAGS) -o test_kgemm_nt_batched test_kgemm_nt_batched.cpp -lcuda

clean:
	touch test_kgemm_nn_batched test_kgemm_nt_batched kgemm_nt_batched.o
	rm test_kgemm_nn_batched test_kgemm_nt_batched *.o
