# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(SEALDemo VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(test log_reg.cpp)

find_package(SEAL 3.6.4 REQUIRED)
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

if(LAPACK_FOUND AND BLAS_FOUND)
   set(lapackblas_libraries ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
   message(here : '${LAPACK_LIBRARIES}, ${BLAS_LIBRARIES}')
endif()

target_link_libraries(test SEAL::seal)
target_link_libraries(test 'armadillo')
target_link_libraries(test 'lapack')