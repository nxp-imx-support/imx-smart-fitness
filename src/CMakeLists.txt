add_subdirectory(classifier)
add_subdirectory(utils)
add_subdirectory(mediapipe)
add_subdirectory(cargs)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(imx-smart-fitness main.cc)
target_link_libraries(imx-smart-fitness
    ${GLIB_LIBRARIES} 
    ${GSTREAMER_LIBRARIES}
    cargs
    classifier
    utils
    mediapipe
    cairo
    gstallocators-1.0
    gstvideo-1.0
    )
