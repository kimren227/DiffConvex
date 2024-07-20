project(cmake_and_cuda LANGUAGES C CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(diff_covnex_python diffconvex.cpp)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(TBB REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(SRCS 
	diff_convex.cpp
	diff_convex.hpp
)
add_subdirectory(qhull)
set(QHULL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/qhull/src/libqhullcpp/" "${CMAKE_CURRENT_SOURCE_DIR}/qhull/src/")
add_library(diff_convex STATIC ${SRCS})
target_include_directories(diff_convex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(diff_convex PUBLIC ${QHULL_INCLUDE_DIRS})
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libigl)
igl_include(copyleft cgal)
target_link_libraries(diff_convex PUBLIC TBB::tbb TBB::tbbmalloc CGAL::CGAL CGAL::CGAL_Core)
target_link_libraries(diff_convex PUBLIC igl_copyleft::core igl_copyleft::cgal)
target_link_libraries(diff_convex PUBLIC qhullcpp qhullstatic_r)

target_link_libraries(diff_covnex_python PUBLIC diff_convex)
install(TARGETS diff_convex DESTINATION ${Python_SITELIB})
install(TARGETS diff_covnex_python DESTINATION ${Python_SITELIB})
set_target_properties(diff_covnex_python PROPERTIES OUTPUT_NAME diff_convex)
