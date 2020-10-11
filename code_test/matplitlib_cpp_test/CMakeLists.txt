cmake_minimum_required(VERSION 3.5)
project (matplitlib_cpp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
IF (WIN32)
	MESSAGE(STATUS "Now is windows")
	#set(PYTHONHOME F:/software/python38)	
	set(PYTHONHOME C:/Users/南澈/AppData/Local/Programs/Python/Python36)
	#set(PYTHONHOME F:/software/python38_x64)
	include_directories(${PYTHONHOME}/include)
	include_directories(${PYTHONHOME}/Lib/site-packages/numpy/core/include)
	link_directories(${PYTHONHOME}/libs)
ELSEIF (UNIX)
	MESSAGE(STATUS "Now is UNIX-like OS's.")
	#find_package(Python3 COMPONENTS Development NumPy)
	# adding...
	find_package(PythonLibs 2.7)
	add_executable(simple_test src/simple_test.cpp)
	target_include_directories(simple_test PRIVATE ${PYTHON_INCLUDE_DIRS})
	target_link_libraries(simple_test ${PYTHON_LIBRARIES})
	#include_directories(/usr/include/python3.5)
	include_directories(/usr/local/lib/python2.7/dist-packages/numpy/core/include)
	#link_directories(/usr/lib/python3.5)
ENDIF ()



include_directories(include)



add_definitions(-DOPJ_STATIC)
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

# message(STATUS "*** dump start cmake variables ***")
# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#         message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
# message(STATUS "*** dump end ***")



