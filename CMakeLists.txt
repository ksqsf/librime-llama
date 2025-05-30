project(rime-llama)
cmake_minimum_required(VERSION 3.10)

aux_source_directory(src llama_src)

add_library(rime-llama-objs OBJECT ${llama_src})
if(BUILD_SHARED_LIBS)
  set_target_properties(rime-llama-objs
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
endif()

# pure CPU, but try to enable as much acceleration as possible
set(LLAMA_METAL OFF CACHE BOOL "Disable Metal" FORCE)
set(LLAMA_NATIVE ON CACHE BOOL "Enable native optimizations" FORCE)
set(LLAMA_ACCELERATE ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "Disable GGML Metal" FORCE)
add_subdirectory(llama.cpp)
include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp/ggml/include")

# librime-llama
set(plugin_name rime-llama PARENT_SCOPE)
set(plugin_objs $<TARGET_OBJECTS:rime-llama-objs> PARENT_SCOPE)
set(plugin_deps ${rime_library} llama PARENT_SCOPE)
set(plugin_modules "llama" PARENT_SCOPE)
