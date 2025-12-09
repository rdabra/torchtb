cmake_minimum_required (VERSION 3.28)


project(torchtb LANGUAGES CXX VERSION 0.1 DESCRIPTION "Helper library for libtorch data prep")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF) #Turns off compiler non-standard features  


#################################################### CUDA Version ####################################################

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Setting CUDA standard to C++20 for Release")
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS ON)
endif()

#################################################### Source Directories ####################################################

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(cmake)


#################################################### Testing Directories ####################################################

enable_testing()
if(BUILD_TESTING)
  add_subdirectory(tests)  
endif()

