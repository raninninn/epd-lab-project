cmake_minimum_required(VERSION 3.10)

# Project name and language (C)
project(LeNet1_C_Model C)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Compiler flags: 
# -Wall -Wextra: Enable strict warnings to catch integer/pointer bugs early
# -O3: Maximum optimization (Crucial for accurate Milestone 1 profiling)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")

# Define the source files
set(SOURCES
    main.c
    src/conv.c
    src/pool.c
    src/fc.c
    src/activation.c
    src/network.c
)

# Create the executable
add_executable(lenet_app ${SOURCES})

# Specify where the compiler should look for header files (.h)
# ${CMAKE_CURRENT_SOURCE_DIR} allows finding "weights_export/..." relative to the root
target_include_directories(lenet_app PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional: Link the math library (libm) if you end up using <math.h> later
# target_link_libraries(lenet_app PRIVATE m)