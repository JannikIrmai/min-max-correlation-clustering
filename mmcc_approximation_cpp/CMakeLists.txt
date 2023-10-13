cmake_minimum_required(VERSION 2.6)
project(min_max_correlation_clustering)

include_directories(include)

file(GLOB headers
    include/*.hxx
)
enable_testing()

add_executable(test-min-max-correlation-clustering src/unit-tests/min_max_correlation_clustering.cxx ${headers})
add_test(test-min-max-correlation-clustering test-min-max-correlation-clustering)

# Python bindings
find_package(pybind11)
if(pybind11_FOUND)
    pybind11_add_module(mmcc_approximation src/python_interface.cxx)
endif()