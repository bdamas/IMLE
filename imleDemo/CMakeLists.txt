# IMLE Demo

# ---------------- REQUIREMENTS -----------------------

set( INCLUDE "")
set( LIBS "")
set( FLAGS "")

# imle
set( INCLUDE ${INCLUDE} ${imle_INCLUDE_DIRS} )
set( LIBS ${LIBS} ${imle_LIBRARIES} )
set( FLAGS "${FLAGS} ${imle_FLAGS}" )


# -------------   DEMO BUILDING   --------------------
#
# IMLE Demonstration Executable
#

set(SRC1 imleDemo.cpp imleDemo_common.hpp)
set(TARGET1 imleDemo)

set(SRC2 imleDemoNoTemplates.cpp imleDemo_common.hpp)
set(TARGET2 imleDemoNoTemplates)

include_directories(${INCLUDE} )

#add_definitions(${FLAGS})
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${FLAGS}" )
SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${FLAGS}" )

add_executable(${TARGET1} ${SRC1})
target_link_libraries(${TARGET1} ${LIBS})

add_executable(${TARGET2} ${SRC2})
target_link_libraries(${TARGET2} ${LIBS})


