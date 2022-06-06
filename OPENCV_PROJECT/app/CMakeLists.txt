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

# for libraries included in the main
#
# target_link_libraries(test_main ......lib_name.......)
#

target_link_libraries( app
    PUBLIC LII::classifier)
    
target_link_libraries( app
    PUBLIC LII::segmentation)
    
target_link_libraries( app
    PUBLIC LII::detection)

target_link_libraries( app
    PUBLIC ${OpenCV_LIBS} )

set( CPACK_PROJECT_NAME ${PROJECT_NAME} )
set( CPACK_PROJECT_VERSION ${PROJECT_VERSION} )
include( CPack )
