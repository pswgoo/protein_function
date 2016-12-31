set(SRCS
    mess_statistic_main.cpp
)
add_source_groups(${SRCS})

add_executable(mess_statistic_main ${SRCS})
target_link_libraries(mess_statistic_main
    data_class
)
