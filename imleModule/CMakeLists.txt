# IMLE module for YARP

# ---------------- REQUIREMENTS -----------------------

# YARP
find_package(YARP REQUIRED)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${YARP_LINK_FLAGS}")              


# -------------  MODULE BUILDING  --------------------

#
# IMLE Yarp Module (server)
#

set(SRC "onlineLearner.cpp" onlineLearnerMain.cpp onlineLearner.hpp)

#option( BUILD_IMLE_MODULE_TEMPLATES "Build Template-based Version of IMLE Module?" ON)
#if( BUILD_IMLE_MODULE_TEMPLATES )
    set(INPUT_DIM "1" CACHE STRING "IMLE Template-based Module Input Dimension")
    set(OUTPUT_DIM "1" CACHE STRING "IMLE Template-based Module Output Dimension")

    set(TARGET onlineLearner${INPUT_DIM}to${OUTPUT_DIM})

    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} imleLib ${YARP_LIBRARIES} )
    target_include_directories(${TARGET} PRIVATE ${YARP_INCLUDE_DIRS} )
    target_compile_definitions(${TARGET} PRIVATE INPUT_DIM=${INPUT_DIM} OUTPUT_DIM=${OUTPUT_DIM})
#else( BUILD_IMLE_MODULE_TEMPLATES )
#    unset(INPUT_DIM CACHE)
#    unset(OUTPUT_DIM CACHE)
#endif( BUILD_IMLE_MODULE_TEMPLATES )

option( BUILD_IMLE_MODULE_NO_TEMPLATES "Build also No-Template Version of IMLE Module?" OFF)
if( BUILD_IMLE_MODULE_NO_TEMPLATES )
    set(TARGET onlineLearner)

    add_executable(${TARGET} ${SRC})
    target_link_libraries(${TARGET} imleLib ${YARP_LIBRARIES} )
    target_include_directories(${TARGET} PRIVATE ${YARP_INCLUDE_DIRS} )
    target_compile_definitions(${TARGET} PRIVATE IMLE_NO_TEMPLATES)
endif( BUILD_IMLE_MODULE_NO_TEMPLATES )


