project(cuhear_tests CUDA)
set(CMAKE_CUDA_STANDARD 14)

file(GLOB sources_cu *.cu)
file(GLOB sources_cpp *.cpp)
add_executable(cuhear_tests ${sources_cu} ${sources_cpp})

target_link_libraries(cuhear_tests PRIVATE cuhear)
set_target_properties(cuhear_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CUDA_FLAGS "--compiler-options -march=native --compiler-options -fopenmp")
set(CMAKE_CXX_FLAGS "-march=native -fopenmp")

set_target_properties(cuhear_tests PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
