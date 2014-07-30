# IMLE Lib

# All warnings?
option(ALL_WARNINGS "Set all compilation warnings on?" ON)

# ------------- LIBRARY BUILDING  --------------------

set( SRC imle.hpp imleInline.hpp expert.hpp expertInline.hpp )
set( SRC ${SRC} imle.cpp)

add_library(imleLib ${SRC})
target_link_libraries(imleLib PUBLIC eigenSerialized)
target_include_directories(eigenSerialized INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} )
if( ALL_WARNINGS )
    target_compile_options(imleLib PUBLIC -ansi -pedantic -Wall)
endif( ALL_WARNINGS )

