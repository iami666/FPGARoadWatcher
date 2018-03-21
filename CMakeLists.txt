cmake_minimum_required (VERSION 2.6)
project (FPGARoadDetect)
find_package( OpenCV REQUIRED )
add_executable(FPGARoadDetect src/extract_vanishing.cpp)
target_link_libraries( FPGARoadDetect ${OpenCV_LIBS} )

