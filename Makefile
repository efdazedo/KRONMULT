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
	kronmult1_pbatched.cpp \
	kronmult1_pbatched.hpp \
	kronmult1.hpp \
	kronmult2_batched.cpp \
	kronmult2_batched.hpp \
	kronmult2_pbatched.cpp \
	kronmult2_pbatched.hpp \
	kronmult2_xbatched.cpp \
	kronmult2_xbatched.hpp \
	kronmult2.hpp \
	kronmult3_batched.cpp \
	kronmult3_batched.hpp \
	kronmult3_pbatched.cpp \
	kronmult3_pbatched.hpp \
	kronmult3_xbatched.cpp \
	kronmult3_xbatched.hpp \
	kronmult3.hpp \
	kronmult4_batched.cpp \
	kronmult4_batched.hpp \
	kronmult4_pbatched.cpp \
	kronmult4_pbatched.hpp \
	kronmult4_xbatched.cpp \
	kronmult4_xbatched.hpp \
	kronmult4.hpp \
	kronmult5_batched.cpp \
	kronmult5_batched.hpp \
	kronmult5_pbatched.cpp \
	kronmult5_pbatched.hpp \
	kronmult5_xbatched.cpp \
	kronmult5_xbatched.hpp \
	kronmult5.hpp \
	kronmult6_batched.cpp \
	kronmult6_batched.hpp \
	kronmult6_pbatched.hpp \
	kronmult6_pbatched.cpp \
	kronmult6_xbatched.hpp \
	kronmult6_xbatched.cpp \
	kronmult6.hpp 

all: test_kgemm_nn_batched test_kgemm_nt_batched test_kronmult6_batched test_kronmult6_pbatched test_kronmult6_xbatched


test_kgemm_nn_batched: test_kgemm_nn_batched.cpp kgemm_nn_batched.hpp kgemm_nn.hpp
	$(CXX) $(CXXFLAGS) -o test_kgemm_nn_batched test_kgemm_nn_batched.cpp $(LIBS) 

test_kgemm_nt_batched: test_kgemm_nt_batched.cpp kgemm_nt_batched.hpp kgemm_nt.hpp
	$(CXX) $(CXXFLAGS) -o test_kgemm_nt_batched test_kgemm_nt_batched.cpp $(LIBS) 

test_kronmult6_batched: test_kronmult6_batched.cpp $(KRONSRC)
	$(CXX) $(CXXFLAGS) -o test_kronmult6_batched test_kronmult6_batched.cpp $(LIBS)

test_kronmult6_pbatched: test_kronmult6_pbatched.cpp $(KRONSRC)
	$(CXX) $(CXXFLAGS) -o test_kronmult6_pbatched test_kronmult6_pbatched.cpp $(LIBS)

test_kronmult6_xbatched: test_kronmult6_xbatched.cpp $(KRONSRC)
	$(CXX) $(CXXFLAGS) -o test_kronmult6_xbatched test_kronmult6_xbatched.cpp $(LIBS)


clean:
	touch test_kronmult6_batched  test_kronmult6_pbatched
	touch test_kronmult6_xbatched
	touch test_kgemm_nn_batched test_kgemm_nt_batched kgemm_nt_batched.o
	rm test_kgemm_nn_batched test_kgemm_nt_batched *.o
	rm test_kronmult6_batched  test_kronmult6_pbatched
	rm test_kronmult6_xbatched
