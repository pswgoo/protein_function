set(SRCS
	unit_test_select_data_main.cpp
)
add_source_groups(${SRCS})

add_executable(unit_test_select_data_main ${SRCS})
target_link_libraries(unit_test_select_data_main
	data_class
)
