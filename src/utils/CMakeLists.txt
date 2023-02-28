set(UTILS_MODULE utils)

aux_source_directory(. UTILS_SOURCE)
add_library(${UTILS_MODULE} STATIC ${UTILS_SOURCE})
