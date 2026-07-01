cmake_minimum_required(VERSION 3.21)

project(VapourSynth-BilateralHIP LANGUAGES CXX HIP)

find_package(hip REQUIRED)

set(VAPOURSYNTH_INCLUDE_DIRECTORY "" CACHE PATH "Path to VapourSynth headers")

if(NOT VAPOURSYNTH_INCLUDE_DIRECTORY)
    find_package(PkgConfig QUIET MODULE)

    if(PKG_CONFIG_FOUND)
        pkg_search_module(VS vapoursynth)

        if(VS_FOUND)
            message(STATUS "Found VapourSynth r${VS_VERSION}")

            set(VAPOURSYNTH_INCLUDE_DIRECTORY ${VS_INCLUDE_DIRS})

            cmake_path(APPEND VS_INSTALL_DIR ${VS_LIBDIR} vapoursynth)
        endif()
    endif()
endif()

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

configure_file(source/config.h.in config.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(bilateralhip_kernel OBJECT source/kernel.hip)

set_target_properties(bilateralhip_kernel PROPERTIES
    HIP_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON)

add_library(bilateralhip_source OBJECT source/bilateral.cpp)

target_include_directories(bilateralhip_source PRIVATE
    ${VAPOURSYNTH_INCLUDE_DIRECTORY})

target_link_libraries(bilateralhip_source PRIVATE hip::host)

set_target_properties(bilateralhip_source PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

add_library(bilateralhip SHARED)
target_link_libraries(bilateralhip bilateralhip_source bilateralhip_kernel hip::host)

if(WIN32)
    install(TARGETS bilateralhip RUNTIME)
elseif(DEFINED VS_INSTALL_DIR)
    install(TARGETS bilateralhip LIBRARY DESTINATION ${VS_INSTALL_DIR})
else()
    install(TARGETS bilateralhip
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
