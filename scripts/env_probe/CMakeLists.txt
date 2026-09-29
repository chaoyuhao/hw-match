cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(cann_environment_probe LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 14)
if(NOT DEFINED NPU_ARCH)
    set(NPU_ARCH "dav-2201")
endif()
add_executable(env_probe env_probe.asc)
# Match the competition template's link requirements as well as its ASC build.
target_link_libraries(env_probe PRIVATE
    tiling_api register platform unified_dlog dl m graph_base
)
target_compile_options(env_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${NPU_ARCH}>
)
