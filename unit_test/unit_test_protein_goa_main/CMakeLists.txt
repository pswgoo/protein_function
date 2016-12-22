set(SRCS
	unit_test_protein_goa_main.cpp
)
add_source_groups(${SRCS})

add_executable(unit_test_protein_goa_main ${SRCS})
target_link_libraries(unit_test_protein_goa_main
	data_class
)
