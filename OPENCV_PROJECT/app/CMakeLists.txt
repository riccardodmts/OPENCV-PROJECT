# ./app/CMakeLists.txt

cmake_minimum_required( VERSION 3.10.2 )

project( app
         VERSION 1.0
         DESCRIPTION "An application to verify the hand detection and segmentation algorithm" )

include( CTest )
enable_testing()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(app src/hand_detection_segmentation.cpp)


target_link_libraries( app
    PUBLIC LII::classifier)
    
target_link_libraries( app
    PUBLIC LII::skin_segmentation)
    
target_link_libraries( app
    PUBLIC LII::detection)

target_link_libraries( app
    PUBLIC LII::evaluation)

target_link_libraries( app
    PUBLIC LII::segment)

target_link_libraries( app
    PUBLIC ${OpenCV_LIBS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )
