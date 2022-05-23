cmake_minimum_required(VERSION 3.5)

project(fem_lib)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions("-Wall")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# OpenMP
# if(NOT TARGET OpenMP::OpenMP_CXX)
#     find_package(Threads REQUIRED)
#     add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
#     set_property(TARGET OpenMP::OpenMP_CXX
#                  PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
#     # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
#     set_property(TARGET OpenMP::OpenMP_CXX
#                  PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
#     add_definitions(-DEIGEN_DONT_PARALLELIZE)
#     message("-- OpenMP is detected.")
#     add_compile_options("-fopenmp")
# endif()
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
#    add_definitions(-DEIGEN_DONT_PARALLELIZE)
    message("OpenMP is detected.")
else()
    message("Multi-process not ready.")
endif()

# Include directories.
include_directories("../external/eigen-3.4.0/")
include_directories("./include")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# The core library.
file(GLOB_RECURSE CORE_CPP "./src/*.cpp")
add_library(fem_lib SHARED ${CORE_CPP})
target_link_libraries(fem_lib PRIVATE OpenMP::OpenMP_CXX)

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./src/*.cpp" "./py_fem_lib_wrap.cxx")
add_library(py_fem_lib SHARED ${CORE_HEADER} ${PYTHON_BINDING})
target_link_libraries(py_fem_lib PRIVATE OpenMP::OpenMP_CXX)
