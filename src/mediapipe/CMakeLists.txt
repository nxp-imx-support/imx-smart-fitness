set(MEDIAPIPE_MODULE mediapipe)

aux_source_directory(. MEDIAPIPE_SOURCE)
add_library(${MEDIAPIPE_MODULE} STATIC ${MEDIAPIPE_SOURCE})
