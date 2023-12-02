cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CUDA_RAY_MESH_INTERSECTION LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")


# 寻找CUDA库（可选，如果需要链接CUDA库）
find_package(CUDA REQUIRED)

# 包含目录
include_directories(${CUDA_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES
     "utils/*.cpp"
)
file(GLOB_RECURSE CUDA_SOURCES
    "intersection001/*.cu"
)


# 指定源文件
# add_executable(RayMeshIntersection main.cpp ${SOURCES} ${CUDA_SOURCES})
add_executable(RayMeshIntersection main.cpp ${SOURCES} ${CUDA_SOURCES})

# 链接CUDA库到你的应用程序
target_link_libraries(RayMeshIntersection ${CUDA_LIBRARIES})
