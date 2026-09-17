# Build de HOST (Linux) del mismo C++ que va al .so, como binario de consola.
# Lo consume tools/parity/parity.py para comparar contra el pipeline Python.
#
#   cmake -S tools/host -B build_host -DCMAKE_BUILD_TYPE=Release && cmake --build build_host
#
# Necesita los prebuilts Linux en third_party/host (tools/host/fetch.sh).
cmake_minimum_required(VERSION 3.22)
project(piuocr_host CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(TP ${ROOT}/third_party/host)
set(SRC ${ROOT}/src/main/cpp)

set(OpenCV_DIR ${TP}/opencv-mobile-4.13.0-ubuntu-2404/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
set(ncnn_DIR ${TP}/ncnn-20240820-ubuntu-2404/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
set_target_properties(ncnn PROPERTIES INTERFACE_COMPILE_OPTIONS "")

add_executable(piuocr_cli cli.cpp
    ${SRC}/pipeline.cpp ${SRC}/detector.cpp ${SRC}/segment.cpp
    ${SRC}/text.cpp ${SRC}/badge.cpp ${SRC}/recognize.cpp ${SRC}/score.cpp
    ${SRC}/rectify.cpp)
target_include_directories(piuocr_cli PRIVATE ${SRC})
target_link_libraries(piuocr_cli ${OpenCV_LIBS} ncnn)
# Mismas flags que el .so: el punto es medir el mismo código.
target_compile_options(piuocr_cli PRIVATE -O3 -ffast-math -fexceptions)
