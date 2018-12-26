cmake_minimum_required(VERSION 3.10.2)

project(yolo_markers)

set(CMAKE_CXX_STANDARD 14)
cmake_policy(SET CMP0022 NEW)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


find_package(Boost
  REQUIRED
  COMPONENTS filesystem program_options
  )
  
set(BOOST_DEFINES "BOOST_THREAD_VERSION=5;BOOST_THREAD_PROVIDES_CONDITION;BOOST_THREAD_PROVIDES_NESTED_LOCKS;BOOST_THREAD_PROVIDES_FUTURE;BOOST_THREAD_PROVIDES_FUTURE_CONTINUATION;BOOST_THREAD_PROVIDES_EXECUTORS;BOOST_ALL_DYN_LINK")

find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

	
#############################################


set(dataset_prepare_SRCS
	dataset_prepare.h
    dataset_prepare.cpp
)

add_executable(dataset_prepare ${dataset_prepare_SRCS})
target_compile_definitions (dataset_prepare PUBLIC ${BOOST_DEFINES})
target_include_directories(dataset_prepare PUBLIC ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries( dataset_prepare ${Boost_LIBRARIES} ${OpenCV_LIBS})

#############################################

