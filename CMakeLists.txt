cmake_minimum_required(VERSION 3.16)
project(Real-time-Object-Recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(objRec objRecognition.cpp features.cpp csv.cpp utilities.cpp)

target_include_directories(objRec PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(objRec PRIVATE ${OpenCV_LIBS})