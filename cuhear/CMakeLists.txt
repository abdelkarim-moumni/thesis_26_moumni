project(cuhear CUDA)
set(CMAKE_CUDA_STANDARD 14)

find_package(MPI REQUIRED)

file(GLOB sources_cu source/*.cu)
file(GLOB sources_cpp source/*.cpp)
add_library(cuhear SHARED ${sources_cu} ${sources_cpp})

set(CMAKE_CUDA_FLAGS "--compiler-options -march=native")
include_directories(cuhear PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(cuhear PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(cuhear PUBLIC ${MPI_INCLUDE_PATH})

target_link_libraries(cuhear PUBLIC MPI::MPI_CXX)

set_target_properties(cuhear PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cuhear PROPERTIES CUDA_RUNTIME_LIBRARY Shared)