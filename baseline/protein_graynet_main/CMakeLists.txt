set(SRCS
	protein_graynet_main.cpp
)
add_source_groups(${SRCS})

add_executable(protein_graynet_main ${SRCS})
target_link_libraries(protein_graynet_main
	data_class
	learning
	graynet
	${Boost_LIBRARIES}
)
