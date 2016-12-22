
set(SRCS
	select_cafa1_dataset_main.cpp
)
add_source_groups(${SRCS})

add_executable(select_cafa1_dataset_main ${SRCS})
target_link_libraries(select_cafa1_dataset_main
	data_class
	${Boost_LIBRARIES}
)
