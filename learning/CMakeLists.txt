set(SRCS
	evaluation.h
	evaluation.cpp
)

add_source_groups(${SRCS})

add_library(learning STATIC ${SRCS})

target_link_libraries(learning
	${Boost_LIBRARIES}
)
