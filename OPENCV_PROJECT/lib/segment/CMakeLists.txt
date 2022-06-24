cmake_minimum_required( VERSION 3.10.2)

project( segment
         VERSION 0.9
         DESCRIPTION "segmentation library" )

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )

add_library( segment 
    STATIC src/segment.cpp)

target_include_directories( segment
    PUBLIC include )
    
target_link_libraries( segment
    PUBLIC ${OpenCV_LIBS} )

add_library( LII::segment 
    ALIAS segment )
