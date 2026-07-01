cmake_minimum_required(VERSION 3.21)

project(VapourSynth-BilateralHIP LANGUAGES CXX HIP)

find_package(hip REQUIRED)

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP ${VCS_TAG} VCS_TAG)
    endif()
endif()

if(VCS_TAG)
    message(STATUS "VapourSynth-BilateralHIP ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

if(WIN32)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/manifest.vs" "[VapourSynth Manifest V1]\nbilateralhip\n")
else()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/manifest.vs" "[VapourSynth Manifest V1]\nlibbilateralhip\n")
endif()

configure_file(source/config.h.in config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(bilateralhip_kernel OBJECT source/kernel.hip)

set_target_properties(bilateralhip_kernel PROPERTIES
    HIP_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)

add_library(bilateralhip_source OBJECT source/bilateral.cpp)


target_link_libraries(bilateralhip_source PRIVATE hip::host)

set_target_properties(bilateralhip_source PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_library(bilateralhip SHARED)
target_link_libraries(bilateralhip bilateralhip_source bilateralhip_kernel hip::host)

find_package(PkgConfig QUIET MODULE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)
    if(VS_FOUND)
        message(STATUS "Found VapourSynth R${VS_VERSION}")
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import vapoursynth; print(vapoursynth.get_plugin_dir())"
            OUTPUT_VARIABLE VS_PLUGIN_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        target_include_directories(bilateralhip_source PRIVATE ${VS_INCLUDE_DIRS})
        install(TARGETS bilateralhip LIBRARY DESTINATION "${VS_PLUGIN_DIR}/bilateralhip")
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/manifest.vs" DESTINATION "${VS_PLUGIN_DIR}/bilateralhip")
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")

    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(bilateralhip PRIVATE ${VS_INCLUDE_DIR})

    install(TARGETS bilateralhip LIBRARY RUNTIME)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/manifest.vs" DESTINATION .)
endif()

if(WIN32)
  file(TO_CMAKE_PATH "$ENV{HIP_PATH}" HIP_ROOT_DIR)
  if(NOT HIP_ROOT_DIR)
    file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" HIP_ROOT_DIR)
  endif()

  if(HIP_ROOT_DIR)
    set(HIP_BIN_DIR "${HIP_ROOT_DIR}/bin")
    file(GLOB _DLLS "${HIP_BIN_DIR}/amdhip64_*.dll")

    if(_DLLS)
      install(FILES ${_DLLS} DESTINATION bin)
    else()
      message(WARNING "Could not find HIP DLLs in ${HIP_BIN_DIR}")
    endif()
  endif()
endif()
