cmake_minimum_required(VERSION 2.7)
project(lab5)

find_package(MPI)

set(VT_LIBRARIES "")
if (${CMAKE_CXX_COMPILER} MATCHES "mpiicpc.*$")
    set(MPI_COMP_FLAGS -Nmpi)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -tcollect -trace -I$ENV{VT_ROOT}/include")
    set(VT_LIBRARIES "-L$ENV{VT_LIB_DIR} -lVT $ENV{VT_ADD_LIBS}")
else ()
    set(MPI_COMP_FLAGS "")
endif ()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror  -g -std=c++11")

set(SOURCE_FILES main.cpp physics.cpp mpi_env.cpp)
add_executable(lab5 ${SOURCE_FILES})

target_link_libraries(lab5 ${MPI_LIBRARIES} ${VT_LIBRARIES})
