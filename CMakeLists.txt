cmake_minimum_required(VERSION 3.10)
project(cheburaxa VERSION 0.0.1 LANGUAGES CXX)

option(BUILD_EXAMPLE "Build example" OFF)

set(CMAKE_CXX_STANDARD_REQUIRED OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) \
and run CMake from there."
  )
endif()

if (NOT CMAKE_BUILD_TYPE) # if CMAKE_BUILD_TYPE is undefined
    set(CMAKE_BUILD_TYPE Release) # set it to Release
endif()

include(GNUInstallDirs)

add_library(cheburaxa INTERFACE)
target_include_directories(cheburaxa INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>)

if (BUILD_EXAMPLE)
    add_executable(cheburaxa_example examples/example.cpp)
    target_link_libraries(cheburaxa_example PRIVATE cheburaxa)
endif (BUILD_EXAMPLE)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


install(
        TARGETS cheburaxa
        EXPORT cheburaxaTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(cheburaxa_CMAKE_PACKAGE_INSTALL_SUBDIR "share/cheburaxa/cmake")

install(
        EXPORT cheburaxaTargets
        NAMESPACE CHEBURAXA::
        DESTINATION ${cheburaxa_CMAKE_PACKAGE_INSTALL_SUBDIR})

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
                                 ${CMAKE_CURRENT_BINARY_DIR}/cheburaxaConfigVersion.cmake
                                 VERSION ${PROJECT_VERSION}
                                 COMPATIBILITY SameMinorVersion)

configure_package_config_file(
                              "${PROJECT_SOURCE_DIR}/cmake/cheburaxaConfig.cmake.in" ${CMAKE_CURRENT_BINARY_DIR}/cheburaxaConfig.cmake
                              INSTALL_DESTINATION ${cheburaxa_CMAKE_PACKAGE_INSTALL_SUBDIR})

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cheburaxaConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/cheburaxaConfigVersion.cmake"
        DESTINATION ${cheburaxa_CMAKE_PACKAGE_INSTALL_SUBDIR})