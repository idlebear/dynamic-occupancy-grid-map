
SET(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "--shared --compiler-options '-fPIC'" )

FIND_PACKAGE( CUDA REQUIRED )

ADD_EXECUTABLE (dogm_spec dogm_spec.cpp)
TARGET_LINK_LIBRARIES (
    dogm_spec
    PRIVATE dogm
    PRIVATE gtest_main
    ${CUDA_LIBRARIES})

ADD_GTEST(dogm_spec)
