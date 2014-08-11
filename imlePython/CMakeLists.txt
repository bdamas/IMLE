# IMLE python bindings

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/imlePython/cmake ${CMAKE_MODULE_PATH})


FIND_PACKAGE(PythonLibs REQUIRED)
find_package(NumPy REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

IF(Boost_FOUND)
    INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "${PYTHON_INCLUDE_DIRS}" "${NUMPY_INCLUDE_DIRS}")

    SET(Boost_USE_STATIC_LIBS OFF)
    SET(BOOST_ALL_DYN_LINK ON)
    SET(Boost_USE_MULTITHREADED ON)
    SET(Boost_USE_STATIC_RUNTIME OFF)

    FIND_PACKAGE(Boost COMPONENTS python)

    set_source_files_properties(eigen_to_numpy.cpp PROPERTIES COMPILE_FLAGS -std=c++0x)

    ADD_LIBRARY(imle SHARED imle_pythonmod.cpp imlePython.cpp eigen_to_numpy.cpp eigen_to_list.cpp)
    TARGET_LINK_LIBRARIES(imle ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} imleLib)
    SET_TARGET_PROPERTIES(imle PROPERTIES OUTPUT_NAME "_imle" PREFIX "" SUFFIX ".so")
    install(TARGETS imle LIBRARY DESTINATION lib)
    install(CODE "execute_process(COMMAND python setup.py install -f
--prefix=${CMAKE_INSTALL_PREFIX} WORKING_DIRECTORY ../imlePython/)")

ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")

ENDIF()
