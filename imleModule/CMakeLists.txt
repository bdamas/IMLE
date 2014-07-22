# IMLE module for YARP

# ---------------- REQUIREMENTS -----------------------

set( INCLUDE "")
set( LIBS "")
set( FLAGS "")

# imle
set( INCLUDE ${INCLUDE} ${imle_INCLUDE_DIRS} )
set( LIBS ${LIBS} ${imle_LIBRARIES} )
set( FLAGS "${FLAGS} ${imle_FLAGS}" )

# YARP
find_package(YARP REQUIRED)
set( INCLUDE ${INCLUDE} ${YARP_INCLUDE_DIRS})
set( LIBS ${LIBS} ${YARP_LIBRARIES})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${YARP_LINK_FLAGS}")              


# -------------  MODULE BUILDING  --------------------

option( BUILD_IMLE_MODULE_NO_TEMPLATES "Build No-Template Version of IMLE Module?" OFF)
if( BUILD_IMLE_MODULE_NO_TEMPLATES )
    unset(INPUT_DIM CACHE)
    unset(OUTPUT_DIM CACHE)
    add_definitions("-D IMLE_NO_TEMPLATES")
else( BUILD_IMLE_MODULE_NO_TEMPLATES )
    set(INPUT_DIM "1" CACHE STRING "Input Dimension")
    set(OUTPUT_DIM "1" CACHE STRING "Output Dimension")
    add_definitions("-D INPUT_DIM=${INPUT_DIM} -D OUTPUT_DIM=${OUTPUT_DIM}")
endif( BUILD_IMLE_MODULE_NO_TEMPLATES )

#
# IMLE Yarp Module (server)
#

set(SRC "onlineLearner.cpp" onlineLearnerMain.cpp onlineLearner.hpp)
if( BUILD_IMLE_MODULE_NO_TEMPLATES )
    set(TARGET onlineLearner)
else( BUILD_IMLE_MODULE_NO_TEMPLATES )
    set(TARGET onlineLearner${INPUT_DIM}to${OUTPUT_DIM})
endif( BUILD_IMLE_MODULE_NO_TEMPLATES )

include_directories(${INCLUDE})

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})


