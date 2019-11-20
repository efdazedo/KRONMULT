CXXFLAGS = -DUSE_GPU
NVCC=nvcc -x cu 

test_kgemm_nn_batched: test_kgemm_nn_batched.cpp kgemm_nn_batched.hpp kgemm_nn.hpp
	$(NVCC) $(CXXFLAGS) -o test_kgemm_nn_batched test_kgemm_nn_batched.cpp -lcuda

clean:
	rm *.o test_kgemm_nn_batched
