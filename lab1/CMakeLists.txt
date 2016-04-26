cmake_minimum_required(VERSION 2.8)
project(lab1)

find_package(MPI)

if (${CMAKE_C_COMPILER} MATCHES "icc.*$")
    set(MPI_COMP_FLAGS -Nmpi)
else ()
    set(MPI_COMP_FLAGS "")
endif ()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror ${MPI_COMP_FLAGS} -std=c99")

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/filters)
# include_directories(filters)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

set(SOURCE_FILES types.c mpi_env.c filters/ppmio.c)
set(BLUR_FILES blur_main.c blur_filter.c filters/gaussw.c)
set(THRE_FILES threshold_main.c threshold_filter.c blur_filter.c)

set(COMMON_LIBS m pthread)

add_executable(lab1_blur ${BLUR_FILES} ${SOURCE_FILES})
add_executable(lab1_thre ${THRE_FILES} ${SOURCE_FILES})
target_link_libraries(lab1_blur ${MPI_C_LIBRARIES} ${COMMON_LIBS})
target_link_libraries(lab1_thre ${MPI_C_LIBRARIES} ${COMMON_LIBS})
