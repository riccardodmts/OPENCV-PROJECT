cmake_minimum_required( VERSION 3.10.2 )

project( Create_and_test_hand_detection_and_segmentation_library
         VERSION 0.9
         DESCRIPTION "Implementation of hand detection and segmentation" )

#if (MSVC)
    # warning level 4 and all warnings as errors
    #add_compile_options(/W4 /WX)
#else()
    # lots of warnings and all warnings as errors
    #add_compile_options(-Wall -Wextra -pedantic)
#endif()

# Create targets for building the (local) libraries.
add_subdirectory( lib/classifier )
add_subdirectory( lib/detection )
add_subdirectory( lib/skin_segmentation )
add_subdirectory( lib/evaluation )
add_subdirectory( lib/segment )

# Create the targets for the entire example-app.
add_subdirectory( app )
