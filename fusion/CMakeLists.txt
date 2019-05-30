cmake_minimum_required(VERSION 2.8)
project(fusion)

set( CMAKE_CXX_FLAGS "--std=gnu++11"  ${CMAKE_CXX_FLAGS} )
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
set(CMAKE_BUILD_TYPE "Release")


find_package(OpenCV REQUIRED)
include_directories(include)
include_directories(/home/lhc/library/eigen-eigen-5a0156e40feb/)

add_executable(fusion src/main.cpp src/file_io.cpp src/pro_and_fusion.cpp src/utility.cpp)


target_link_libraries(fusion ${OpenCV_LIBS})