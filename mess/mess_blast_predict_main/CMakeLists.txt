set(SRCS
        mess_blast_predict_main.cpp
)
add_source_groups(${SRCS})

add_executable(mess_blast_predict_main ${SRCS})
target_link_libraries(mess_blast_predict_main 
        data_class
		learning
)
