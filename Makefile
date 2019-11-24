include make.inc

KRONSRC=  \
	kgemm_nn_batched.cpp \
	kgemm_nn_batched.hpp \
	kgemm_nn.cpp \
	kgemm_nn.hpp \
	kgemm_nt_batched.cpp \
	kgemm_nt_batched.hpp \
	kgemm_nt.cpp \
	kgemm_nt.hpp \
	kroncommon.hpp \
	kronmult1_batched.cpp \
	kronmult1_batched.hpp \
	kronmult1.hpp \
	kronmult2_batched.cpp \
	kronmult2_batched.hpp \
	kronmult2.hpp \
	kronmult3_batched.cpp \
	kronmult3_batched.hpp \
	kronmult3.hpp \
	kronmult4_batched.cpp \
	kronmult4_batched.hpp \
	kronmult4.hpp \
	kronmult5_batched.cpp \
	kronmult5_batched.hpp \
	kronmult5.hpp \
	kronmult6_batched.cpp \
	kronmult6_batched.hpp \
	kronmult6.hpp 

all: test_kgemm_nn_batched test_kgemm_nt_batched test_kronmult6_batched


test_kgemm_nn_batched: test_kgemm_nn_batched.cpp kgemm_nn_batched.hpp kgemm_nn.hpp
	$(CXX) $(CXXFLAGS) -o test_kgemm_nn_batched test_kgemm_nn_batched.cpp -lcuda

test_kgemm_nt_batched: test_kgemm_nt_batched.cpp kgemm_nt_batched.hpp kgemm_nt.hpp
	$(CXX) $(CXXFLAGS) -o test_kgemm_nt_batched test_kgemm_nt_batched.cpp -lcuda

test_kronmult6_batched: test_kronmult6_batched.cpp $(KRONSRC)
	$(CXX) $(CXXFLAGS) -o test_kronmult6_batched test_kronmult6_batched.cpp

clean:
	touch test_kronmult6_batched 
	touch test_kgemm_nn_batched test_kgemm_nt_batched kgemm_nt_batched.o
	rm test_kgemm_nn_batched test_kgemm_nt_batched *.o
	rm test_kronmult6_batched 
