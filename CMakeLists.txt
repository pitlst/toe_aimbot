#--------------------------------------------------------------------------------------#
#   项目总体设置
#--------------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.12)
# 设置项目名称
set(PROJECT_NAME toe_aimbot)
# 在这里直接指定项目为c++项目，节约查找tool chain的时间
project(${PROJECT_NAME} VERSION 1.0.0 LANGUAGES CXX)
# 设置c++标准等级和编译后输出路径
set(CMAKE_CXX_STANDARD 11)
# 导出执行命令到json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 强制化c++标准，当标准不符合要求时强制终止编译而不是采用更旧的标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# 设置编译优化的等级
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# 判断项目平台
if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    message(STATUS ${CMAKE_HOST_SYSTEM_NAME} " 检测到为linux平台")
else()
	message(FATAL_ERROR ${CMAKE_HOST_SYSTEM_NAME} " 检测到不是linux平台,该平台目前未支持") 
endif()

#--------------------------------------------------------------------------------------#
#   项目依赖设置
#--------------------------------------------------------------------------------------#
# 找到海康动态库，保存到变量里
FILE (GLOB HIK_LIBS "/opt/MVS/lib/64/*.so")
# 找到海康头文件，保存到变量里
FILE (GLOB HIK_INCLUDE "/opt/MVS/include/*.h")
include_directories(/opt/MVS/include)
# 找到boost
find_package(Boost REQUIRED)
# 找到opencv
set(OpenCV_DIR /home/cheakf/opencv-4.8.0/build/)
find_package(OpenCV REQUIRED)
# 找到openvino
set(OpenVINO_DIR /home/cheakf/openvino/build/)
find_package(OpenVINO REQUIRED)
message(STATUS "已找到openvino,路径" ${OpenVINO_DIR}/../include)
include_directories(${OpenVINO_DIR}/../include)
include_directories(${OpenVINO_DIR}/../include/ie)
# 设置头文件查找路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
    general
    camera
    serial
    determine
    detect
    detect/intel_nuc_11
    detect/nvidia_orin_nx
)
# 把项目目录定义成宏传递给程序
add_compile_definitions(PROJECT_PATH=\"${PROJECT_SOURCE_DIR}/\")

#--------------------------------------------------------------------------------------#
#   项目编译设置
#--------------------------------------------------------------------------------------#
add_executable(${PROJECT_NAME} 
                ${HIK_INCLUDE}
                debug.hpp
                # general/toe_json.hpp
                # general/toe_structs.hpp
                # general/general.hpp
                # camera/grab_img.hpp
                # camera/grab_img.cpp
                # serial/toe_serial.hpp
                # serial/toe_serial.cpp 
                # determine/determine.hpp
                # determine/determine.cpp
                # detect/detect.hpp
                # detect/detect.cpp
                # detect/intel_nuc_11/ov_detect.hpp
                # detect/intel_nuc_11/ov_detect.cpp
                # detect/nvidia_orin_nx/orin_detect.hpp
                # detect/nvidia_orin_nx/orin_detect.hpp
                main.cpp 
                )
target_link_libraries(${PROJECT_NAME} PRIVATE pthread ${OpenCV_LIBS} ${HIK_LIBS} ${Boost_LIBRARIES} ${TENSORRT_LIBRARY} ${CUDA_LIBRARIES} openvino::runtime)