cmake_minimum_required(VERSION 3.5)
project(lab1)

find_package(MPI)

set(CMAKE_SHARED_LINKER_FLAGS "-lpthread, -lrt, -lm")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -std=c99")

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/filters)
include_directories(filters)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(SOURCE_FILES "main.c" types.c blur_filter.c "filters/ppm.c" "filters/ppmio.c" "filters/gaussw.c" "filters/blurfilter.c" shared_com.c)

add_executable(lab1 ${SOURCE_FILES})
target_link_libraries(lab1 ${MPI_C_LIBRARIES})