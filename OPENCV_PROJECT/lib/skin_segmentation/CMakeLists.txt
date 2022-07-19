cmake_minimum_required( VERSION 3.10.2)

project( skin_segmentation
         VERSION 0.9
         DESCRIPTION "skin segmentation library" )

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )

add_library( skin_segmentation 
    STATIC src/skin_segmentation.cpp)

target_include_directories( skin_segmentation
    PUBLIC include )
    
target_link_libraries( skin_segmentation 
    PUBLIC ${OpenCV_LIBS} )

add_library( LII::skin_segmentation 
    ALIAS skin_segmentation )
