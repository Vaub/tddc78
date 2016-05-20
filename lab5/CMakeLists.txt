cmake_minimum_required(VERSION 3.5)
project(lab5)

find_package(MPI)

if (${CMAKE_CXX_COMPILER} MATCHES "icpc.*$")
    set(MPI_COMP_FLAGS -Nmpi)
else ()
    set(MPI_COMP_FLAGS "")
endif ()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror ${MPI_COMP_FLAGS} -g -std=c++11")

set(SOURCE_FILES main.cpp coordinate.h definitions.h physics.cpp mpi_env.cpp)
add_executable(lab5 ${SOURCE_FILES})

target_link_libraries(lab5 ${MPI_LIBRARIES})
