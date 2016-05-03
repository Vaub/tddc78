cmake_minimum_required(VERSION 2.8)
project(lab2)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -std=c99")

set(SOURCE_FILES filters/gaussw.c filters/ppmio.c image.c pthread_barrier.c)
add_executable(lab2_blur blur_main.c ${SOURCE_FILES})
add_executable(lab2_thre thre_main.c ${SOURCE_FILES})

set(COMMON_LIBS pthread m)

target_link_libraries(lab2_blur ${COMMON_LIBS})
target_link_libraries(lab2_thre ${COMMON_LIBS})
