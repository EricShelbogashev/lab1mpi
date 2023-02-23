cmake_minimum_required(VERSION 3.16)
project(lab1 C)

set(CMAKE_C_STANDARD 99)
add_executable(lab1 main.c)

include_directories(/usr/lib/x86_64-linux-gnu/mpich/include)
include_directories(/usr/local/include/mpe.h)

set(CMAKE_CXX_COMPILER "/usr/bin/mpicxx")

