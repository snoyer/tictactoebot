cmake_minimum_required(VERSION 2.8)
project( test1 )
find_package( OpenCV REQUIRED )
add_executable( demo demo.cpp )
set(CMAKE_CXX_FLAGS "-Wall")
target_link_libraries( demo ${OpenCV_LIBS} )

