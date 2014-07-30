# EigenSerialized

project(eigenSerialized)

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)


# ---------------- REQUIREMENTS -----------------------

# BOOST Serialization
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization)

# EIGEN3
FIND_PACKAGE(Eigen3 REQUIRED)

# ------------- LIBRARY BUILDING  --------------------

# # This is cleaner, but is available only for cmake 3.0. Also, how do I add headers to the IDE with this command?
#add_library(eigenSerialized INTERFACE)

#target_include_directories(eigenSerialized INTERFACE ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
#target_link_libraries(eigenSerialized INTERFACE ${Boost_LIBRARIES})

set( SRC EigenSerialized.hpp EigenMatrixBaseSerialize.hpp)
set(SRC ${SRC} EigenSerializedDummy.cpp)
add_library(eigenSerialized ${SRC})

target_include_directories(eigenSerialized PUBLIC ${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR} )
target_link_libraries(eigenSerialized PUBLIC ${Boost_LIBRARIES})

