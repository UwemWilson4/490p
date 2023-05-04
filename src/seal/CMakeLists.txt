# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(SEALDemo VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(test log_reg.cpp)
add_executable(seal_tests seal_tests.cpp)
add_executable(example seal_example.cpp)

find_package(SEAL 3.6.4 REQUIRED)
find_package(LAPACK REQUIRED)

target_link_libraries(test SEAL::seal)
target_link_libraries(test 'armadillo')
target_link_libraries(test 'lapack')

target_link_libraries(seal_tests SEAL::seal)

target_link_libraries(example SEAL::seal)