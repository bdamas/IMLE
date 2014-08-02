# IMLE module demo

# ---------------- REQUIREMENTS -----------------------

# YARP
find_package(YARP REQUIRED)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${YARP_LINK_FLAGS}")              


# -------------   DEMO BUILDING   --------------------

#
# IMLE Yarp Demonstration Training Module
#
set(SRC onlineLearnerUpdate.cpp)
set(TARGET onlineLearnerUpdate)

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} imleLib ${YARP_LIBRARIES})
target_include_directories(${TARGET} PRIVATE ${YARP_INCLUDE_DIRS})

#
# IMLE Yarp Demonstration Query Module 1
#
set(SRC onlineLearnerQuery1.cpp)
set(TARGET onlineLearnerQuery1)

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} imleLib ${YARP_LIBRARIES})
target_include_directories(${TARGET} PRIVATE ${YARP_INCLUDE_DIRS})

#
# IMLE Yarp Demonstration Query Module 2
#
set(SRC onlineLearnerQuery2.cpp)
set(TARGET onlineLearnerQuery2)

add_executable(${TARGET} ${SRC})
target_link_libraries(${TARGET} imleLib ${YARP_LIBRARIES})
target_include_directories(${TARGET} PRIVATE ${YARP_INCLUDE_DIRS})

#
# Conf files for demonstration
#
file(GLOB ConfigFiles ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.ini)
add_custom_target(copy)
foreach(ConfigFile ${ConfigFiles})
    add_custom_command(
        TARGET onlineLearnerUpdate 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ConfigFile} $<TARGET_FILE_DIR:onlineLearnerUpdate>)
endforeach()

