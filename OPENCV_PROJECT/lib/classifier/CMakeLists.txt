
cmake_minimum_required( VERSION 3.10.2)

project( classifier
         VERSION 0.9
         DESCRIPTION "classifier library." )

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )

add_library( classifier 
    STATIC src/classifier.cpp)

target_include_directories( classifier
    PUBLIC include )
    
target_link_libraries( classifier 
    PUBLIC ${OpenCV_LIBS} )

add_library( LII::classifier 
    ALIAS classifier )
