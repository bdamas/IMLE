# EigenSerialized

project(eigenSerialized)

cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)


# ---------------- REQUIREMENTS -----------------------

set( INCLUDE "")
set( LIBS "")
set( FLAGS "")

# BOOST Serialization
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization)

set( INCLUDE ${INCLUDE} ${Boost_INCLUDE_DIRS})
set( LIBS ${LIBS} ${Boost_LIBRARIES})

# EIGEN3
FIND_PACKAGE(Eigen3 REQUIRED)
set( INCLUDE ${INCLUDE} ${EIGEN3_INCLUDE_DIRS})


# This is for including EigenMatrixBaseSerialize.h
set( INCLUDE ${INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR})

option(ALL_WARNINGS "Set all warnings on?" OFF)
if( ALL_WARNINGS )
    set( FLAGS "-ansi -pedantic -Wall")
endif( ALL_WARNINGS )
#set( FLAGS "${FLAGS} -msse2 -msse3 -fopenmp " )
set( FLAGS "${FLAGS} -msse2 -msse3 -msse4 -fopenmp " )

# ------------- LIBRARY BUILDING  --------------------

## Not working with CodeBlocks and cmake 2.8.3...
## Template only library: build an empty target, so that source files appear in IDE's
#set( SRC EigenSerialized.h EigenMatrixBaseSerialize.h )
#set( TARGET eigenSerialized)
#add_custom_target( ${TARGET} SOURCES ${SRC} )

# Temp solution while the above solution does not work...
set( SRC EigenSerialized.hpp EigenMatrixBaseSerialize.hpp)
set(SRC ${SRC} EigenSerializedDummy.cpp)
set( TARGET eigenSerialized)

include_directories( ${INCLUDE} )

add_library(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})

# ----------------- FORWARD ---------------------------

set(eigenSerialized_INCLUDE_DIRS ${INCLUDE} CACHE INTERNAL "eigenSerialized files include location")
set(eigenSerialized_LIBRARIES ${LIBS} CACHE INTERNAL "eigenSerialized libraries")
set(eigenSerialized_FLAGS ${FLAGS} CACHE INTERNAL "eigenSerialized compilation flags")
set(eigenSerialized_FOUND TRUE CACHE INTERNAL "eigenSerialized libraries and headers found!")



