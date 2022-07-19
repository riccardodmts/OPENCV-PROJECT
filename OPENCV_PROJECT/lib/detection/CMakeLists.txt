cmake_minimum_required( VERSION 3.10.2)

project( detection
         VERSION 0.9
         DESCRIPTION "detection library")

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )

add_library( detection 
    STATIC src/detection.cpp)

target_include_directories( detection 
    PUBLIC include)

target_link_libraries( detection
    PUBLIC LII::detection)
	
    
target_link_libraries( detection 
    PUBLIC ${OpenCV_LIBS} )

add_library( LII::detection 
    ALIAS detection)

