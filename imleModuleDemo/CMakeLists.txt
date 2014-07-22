# IMLE module demo

# ---------------- REQUIREMENTS -----------------------

set( INCLUDE "")
set( LIBS "")
set( FLAGS "")

# YARP
find_package(YARP REQUIRED)
set( INCLUDE ${INCLUDE} ${YARP_INCLUDE_DIRS})
set( LIBS ${LIBS} ${YARP_LIBRARIES})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${YARP_LINK_FLAGS}")              


# -------------   DEMO BUILDING   --------------------

add_definitions("-D INPUT_DIM=${INPUT_DIM} -D OUTPUT_DIM=${OUTPUT_DIM}")

#
# IMLE Yarp Demonstration Training Module
#
set(SRC onlineLearnerUpdate.cpp)
set(TARGET onlineLearnerUpdate)

include_directories(${INCLUDE})

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})

#
# IMLE Yarp Demonstration Query Module 1
#
set(SRC onlineLearnerQuery1.cpp)
set(TARGET onlineLearnerQuery1)

include_directories(${INCLUDE})

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})

#
# IMLE Yarp Demonstration Query Module 2
#
set(SRC onlineLearnerQuery2.cpp)
set(TARGET onlineLearnerQuery2)

include_directories(${INCLUDE})

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})

