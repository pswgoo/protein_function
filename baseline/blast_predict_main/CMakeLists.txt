set(SRCS
	blast_predict_main.cpp
)
add_source_groups(${SRCS})

add_executable(blast_predict_main ${SRCS})
target_link_libraries(blast_predict_main
	data_class
	learning
	${Boost_LIBRARIES}
)
