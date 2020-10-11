cmake_minimum_required(VERSION 3.7)
project (matplitlib_cpp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
IF (WIN32)
	MESSAGE(STATUS "Now is windows")
	#set(PYTHONHOME F:/software/python38)	
	set(PYTHONHOME C:/Users/南澈/AppData/Local/Programs/Python/Python36)
	#set(PYTHONHOME F:/software/python38_x64)
ELSEIF (UNIX)
	MESSAGE(STATUS "Now is UNIX-like OS's.")
	# adding...
ENDIF ()



include_directories(include)
include_directories(${PYTHONHOME}/include)
include_directories(${PYTHONHOME}/Lib/site-packages/numpy/core/include)
link_directories(${PYTHONHOME}/libs)

add_definitions(-DOPJ_STATIC)
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

# message(STATUS "*** dump start cmake variables ***")
# get_cmake_property(_variableNames VARIABLES)
# foreach(_variableName ${_variableNames})
#         message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()
# message(STATUS "*** dump end ***")

add_executable(simple_test src/simple_test.cpp)

