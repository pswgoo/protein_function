set(SRCS
        mess_parse_go_data_main.cpp
)
add_source_groups(${SRCS})

add_executable(mess_parse_go_data_main ${SRCS})
target_link_libraries(mess_parse_go_data_main
        data_class
)
