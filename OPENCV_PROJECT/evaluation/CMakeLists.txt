cmake_minimum_required( VERSION 3.10.2)

project( evaluation
         VERSION 0.9
         DESCRIPTION "segmentation library" )

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )

add_library( evaluation 
    STATIC src/evaluation.cpp)

target_include_directories( evaluation
    PUBLIC include )

    
target_link_libraries( evaluation 
    PUBLIC ${OpenCV_LIBS} )

add_library( LII::evaluation 
    ALIAS evaluation )
