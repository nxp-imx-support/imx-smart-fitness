set(CLASSIFIER classifier)

aux_source_directory(. CLASSIFIER_SOURCE)
add_library(${CLASSIFIER} STATIC ${CLASSIFIER_SOURCE})
