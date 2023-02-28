set(CARGS cargs)

aux_source_directory(. CARGS_SOURCE)
add_library(${CARGS} STATIC ${CARGS_SOURCE})
