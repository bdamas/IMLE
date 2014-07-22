# IMLE Lib


# ---------------- REQUIREMENTS -----------------------

set( INCLUDE "")
set( LIBS "")
set( FLAGS "")

# EigenSerialized
set( INCLUDE ${INCLUDE} ${eigenSerialized_INCLUDE_DIRS} )
set( LIBS ${LIBS} ${eigenSerialized_LIBRARIES} )
set( FLAGS "${FLAGS} ${eigenSerialized_FLAGS}" )

option(BUILD_IMLE_LIB_NO_TEMPLATE "Build IMLE library (no templates)?" ON)


# ------------- LIBRARY BUILDING  --------------------

set( SRC imle.hpp imleInline.hpp expert.hpp expertInline.hpp )
if ( BUILD_IMLE_LIB_NO_TEMPLATE )
    set( SRC ${SRC} imle.cpp)
else ( BUILD_IMLE_LIB_NO_TEMPLATE )
    set( SRC ${SRC} imleDummy.cpp)
endif ( BUILD_IMLE_LIB_NO_TEMPLATE )
set( TARGET imleLib )

include_directories( ${INCLUDE} )

add_library(${TARGET} ${SRC})
target_link_libraries(${TARGET} ${LIBS})

# ----------------- FORWARD ---------------------------

set(imle_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE} CACHE INTERNAL "IMLE include location")
if ( BUILD_IMLE_LIB_NO_TEMPLATE )
    set(imle_LIBRARIES ${LIBS} ${TARGET} CACHE INTERNAL "IMLE libraries")
else ( BUILD_IMLE_LIB_NO_TEMPLATE )
    set(imle_LIBRARIES ${LIBS} CACHE INTERNAL "IMLE libraries")
endif ( BUILD_IMLE_LIB_NO_TEMPLATE )
set(imle_FLAGS ${FLAGS} CACHE INTERNAL "IMLE compilation flags")
set(imle_FOUND TRUE CACHE INTERNAL "IMLE Found!")



