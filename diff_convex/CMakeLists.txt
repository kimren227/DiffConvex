cmake_minimum_required(VERSION 3.15..3.26)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(
  differentiable_convex
  VERSION 0.0.1
  LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)

# Define the minimum C++ standard that is required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configuration options
option(BUILD_PYTHON "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" ON)  

add_subdirectory(diff_convex)


