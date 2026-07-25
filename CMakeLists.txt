cmake_minimum_required(VERSION 3.21)

project(VapourSynth-BilateralHIP LANGUAGES CXX HIP)

# Build options
option(ENABLE_HIP "Build the non-RTC HIP plugin" ON)
option(ENABLE_RTC "Build the RTC (Just-In-Time Compilation) plugin" ON)

if(NOT ENABLE_HIP AND NOT ENABLE_RTC)
    message(FATAL_ERROR "At least one of ENABLE_HIP or ENABLE_RTC must be ON")
endif()

# Dependencies
find_package(hip REQUIRED)

if(ENABLE_RTC)
    find_package(hiprtc REQUIRED)
endif()

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --always
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE VCS_TAG
    )
    if(VCS_TAG)
        string(STRIP "${VCS_TAG}" VCS_TAG)
    endif()
endif()

if(VCS_TAG)
    message(STATUS "VapourSynth-BilateralHIP ${VCS_TAG}")
else()
    message(WARNING "unknown plugin version")
    set(VCS_TAG "unknown")
endif()

# Helpers
function(configure_common target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_EXTENSIONS OFF
    )
endfunction()

function(configure_hip_kernel target)
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        HIP_EXTENSIONS OFF
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )
endfunction()

# Non-RTC plugin
if(ENABLE_HIP)
    # Manifest
    if(WIN32)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/hip/manifest.vs" "[VapourSynth Manifest V1]\nbilateralhip\n")
    else()
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/hip/manifest.vs" "[VapourSynth Manifest V1]\nlibbilateralhip\n")
    endif()

    # Kernel object (compiled with hipcc)
    add_library(bilateralhip_kernel OBJECT source/kernel.hip)
    configure_hip_kernel(bilateralhip_kernel)

    # Source object
    add_library(bilateralhip_source OBJECT source/bilateral.cpp)
    configure_common(bilateralhip_source)
    target_link_libraries(bilateralhip_source PRIVATE hip::host)

    # Final shared library
    add_library(bilateralhip SHARED)
    target_sources(bilateralhip PRIVATE $<TARGET_OBJECTS:bilateralhip_source> $<TARGET_OBJECTS:bilateralhip_kernel>)
    target_link_libraries(bilateralhip PRIVATE hip::host)
    configure_common(bilateralhip)
endif()

# RTC plugin
if(ENABLE_RTC)
    # Manifest
    if(WIN32)
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rtc/manifest.vs" "[VapourSynth Manifest V1]\nbilateralhip_rtc\n")
    else()
        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rtc/manifest.vs" "[VapourSynth Manifest V1]\nlibbilateralhip_rtc\n")
    endif()

    # Kernel object (compiled with hipcc, RTC uses .hpp but still needs HIP compilation)
    add_library(bilateralhip_rtc_kernel OBJECT source_rtc/kernel.hpp)
    configure_hip_kernel(bilateralhip_rtc_kernel)

    # Source object
    add_library(bilateralhip_rtc_source OBJECT source_rtc/bilateral.cpp)
    configure_common(bilateralhip_rtc_source)
    target_link_libraries(bilateralhip_rtc_source PRIVATE hip::host hiprtc::hiprtc)

    # Final shared library
    add_library(bilateralhip_rtc SHARED)
    target_sources(bilateralhip_rtc PRIVATE $<TARGET_OBJECTS:bilateralhip_rtc_source> $<TARGET_OBJECTS:bilateralhip_rtc_kernel>)
    target_link_libraries(bilateralhip_rtc PRIVATE hip::host hiprtc::hiprtc)
    configure_common(bilateralhip_rtc)
endif()

# Config header
configure_file(config.h.in config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# VapourSynth discovery & installation
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

        if(ENABLE_HIP)
            target_include_directories(bilateralhip_source PRIVATE ${VS_INCLUDE_DIRS})
            install(TARGETS bilateralhip LIBRARY DESTINATION "${VS_PLUGIN_DIR}/bilateralhip")
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip/manifest.vs" DESTINATION "${VS_PLUGIN_DIR}/bilateralhip")
        endif()

        if(ENABLE_RTC)
            target_include_directories(bilateralhip_rtc_source PRIVATE ${VS_INCLUDE_DIRS})
            install(TARGETS bilateralhip_rtc LIBRARY DESTINATION "${VS_PLUGIN_DIR}/bilateralhip_rtc")
            install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rtc/manifest.vs" DESTINATION "${VS_PLUGIN_DIR}/bilateralhip_rtc")
        endif()
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")
    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    if(ENABLE_HIP)
        target_include_directories(bilateralhip PRIVATE ${VS_INCLUDE_DIR})
        install(TARGETS bilateralhip LIBRARY RUNTIME)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/hip/manifest.vs" DESTINATION .)
    endif()

    if(ENABLE_RTC)
        target_include_directories(bilateralhip_rtc PRIVATE ${VS_INCLUDE_DIR})
        install(TARGETS bilateralhip_rtc LIBRARY RUNTIME)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rtc/manifest.vs" DESTINATION .)
    endif()
endif()

# Windows DLL dependencies
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
