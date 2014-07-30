# IMLE Demo

# -------------   DEMO BUILDING   --------------------
#
# IMLE Demonstration Executable
#

add_executable(imleDemo imleDemo.cpp imleDemo_common.hpp)
target_link_libraries(imleDemo imleLib)

add_executable(imleDemoNoTemplates imleDemoNoTemplates.cpp imleDemo_common.hpp)
target_link_libraries(imleDemoNoTemplates imleLib)


