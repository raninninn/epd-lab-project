cmake_minimum_required(VERSION 3.20)
project(lenet1 C)
set(CMAKE_C_STANDARD 11)

add_executable(lenet1 main.c
        quant.h
        conv.c
        conv.h
        pool.c
        pool.h
        fc.c
        fc.h
        lenet1.c
        lenet1.h
	relu.c
	relu.h)
target_include_directories(lenet1 PRIVATE ../python/export)