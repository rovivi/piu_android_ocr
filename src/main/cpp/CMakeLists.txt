cmake_minimum_required(VERSION 3.22)
project(piuocr CXX)
set(CMAKE_CXX_STANDARD 17)

set(TP ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party)

# opencv-mobile: solo core+imgproc (~2 MB). El SDK completo son ~60 MB y este
# módulo usa un subconjunto chico — ver docs/ARQUITECTURA.md §4.
set(OpenCV_DIR ${TP}/opencv-mobile-4.13.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ncnn SIN Vulkan: el detector corre con use_vulkan_compute=false, así que
# glslang+SPIRV+MachineIndependent (5.3 MB de .a) eran peso muerto.
set(ncnn_DIR ${TP}/ncnn-20240820-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
# ncnn.cmake propaga -fno-exceptions a quien lo linkea (INTERFACE) y pisa
# nuestro -fexceptions. Se quita: solo afecta a cómo se compila este .so,
# libncnn.a ya está compilado.
set_target_properties(ncnn PROPERTIES INTERFACE_COMPILE_OPTIONS "")

add_library(piuocr SHARED
    jni_bridge.cpp pipeline.cpp detector.cpp segment.cpp text.cpp badge.cpp
    recognize.cpp score.cpp rectify.cpp)
target_link_libraries(piuocr ${OpenCV_LIBS} ncnn jnigraphics log)
# -fexceptions: AGP compila con -fno-exceptions; sin esto una cv::Exception
# cruza el JNI y aborta el proceso en vez de caer en el catch de nativeRead.
target_compile_options(piuocr PRIVATE -O3 -ffast-math -fexceptions -fvisibility=hidden
    -fvisibility-inlines-hidden -ffunction-sections -fdata-sections)
# Solo se exportan los Java_* (JNIEXPORT). Sin --exclude-libs el .so exporta
# todo opencv+ncnn estático y --gc-sections no puede tirar nada.
target_link_options(piuocr PRIVATE -Wl,--gc-sections -Wl,--exclude-libs,ALL
    -Wl,--as-needed)
