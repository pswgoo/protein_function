cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

project(protein_function)

include(cmake/AddFolderSubdirectories.cmake)
include(cmake/AddSourceGroups.cmake)

cmake_policy(SET CMP0054 NEW)

set(Boost_USE_STATIC_LIBS ON)
if(MSVC)
find_package(Boost REQUIRED COMPONENTS chrono date_time filesystem iostreams log regex serialization system thread timer zlib)
else()
find_package(Boost REQUIRED COMPONENTS chrono date_time filesystem log regex serialization system thread timer)
endif()
include_directories(${Boost_INCLUDE_DIRS})

# This does not work on Win64
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++0x)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

include_directories(${root_SOURCE_DIR} ./ ./graynet)
link_directories(/usr/local/lib)

add_subdirectory(common)
add_subdirectory(data_class)
add_subdirectory(unit_test)
add_subdirectory(data_processing)
#add_subdirectory(external_library)
add_subdirectory(learning)
add_subdirectory(mess)
add_subdirectory(baseline)
add_subdirectory(graynet)
