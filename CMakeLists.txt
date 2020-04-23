    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    
    if (USE_GPU)
        project(kronmult LANGUAGES CXX CUDA) 
    else ()
        project(kronmult LANGUAGES CXX) 
    endif ()
    
    find_package(OpenMP REQUIRED)

    include(CTest)
     
    add_library(kron STATIC
	kgemm_nn_batched.cpp 
	kgemm_nn_batched.hpp 
	kgemm_nn.cpp 
	kgemm_nn.hpp 
	kgemm_nt_batched.cpp 
	kgemm_nt_batched.hpp 
	kgemm_nt.cpp 
	kgemm_nt.hpp 
	kroncommon.hpp 
	kronmult1_batched.cpp 
	kronmult1_batched.hpp 
	kronmult1_pbatched.cpp 
	kronmult1_pbatched.hpp 
	kronmult1.hpp 
	kronmult2_batched.cpp 
	kronmult2_batched.hpp 
	kronmult2_pbatched.cpp 
	kronmult2_pbatched.hpp 
	kronmult2.hpp 
	kronmult3_batched.cpp 
	kronmult3_batched.hpp 
	kronmult3_pbatched.cpp 
	kronmult3_pbatched.hpp 
	kronmult3.hpp 
	kronmult4_batched.cpp 
	kronmult4_batched.hpp 
	kronmult4_pbatched.cpp 
	kronmult4_pbatched.hpp 
	kronmult4.hpp 
	kronmult5_batched.cpp 
	kronmult5_batched.hpp 
	kronmult5_pbatched.cpp 
	kronmult5_pbatched.hpp 
	kronmult5.hpp 
	kronmult6_batched.cpp 
	kronmult6_batched.hpp 
	kronmult6_pbatched.hpp 
	kronmult6_pbatched.cpp 
	kronmult6.hpp 
      )
     
    target_compile_features(kron PUBLIC cxx_std_11)
    
    
    if (USE_GPU)
        set_target_properties( kron
                               PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif ()

   
    add_executable(kgemm_nn_test test_kgemm_nn_batched.cpp) 
    add_executable(kgemm_nt_test test_kgemm_nt_batched.cpp)
    add_executable(test_kronmult6_batched test_kronmult6_batched.cpp)
    add_executable(test_kronmult6_pbatched test_kronmult6_pbatched.cpp)
    
    if (USE_GPU)
        set_property(TARGET kgemm_nn_test 
                     PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET kgemm_nt_test 
                     PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET test_kronmult6_batched
                     PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET test_kronmult6_pbatched
                     PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    endif()
    target_link_libraries(kgemm_nn_test PRIVATE kron) 
    target_link_libraries(kgemm_nt_test PRIVATE kron)
    target_link_libraries(test_kronmult6_batched PRIVATE kron)
    target_link_libraries(test_kronmult6_pbatched PRIVATE kron)
    

    if(APPLE AND USE_GPU)
      set_property(TARGET kgemm_nn_test
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
      set_property(TARGET kgemm_nt_test
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
      set_property(TARGET test_kronmult6_batched
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

      set_property(TARGET test_kronmult6_pbatched
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    endif()
