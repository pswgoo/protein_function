set(data_class_SRCS
  protein_sequence.cpp
  protein_sequence.h
  go_term.cpp
  go_term.h
)

add_source_groups(${data_class_SRCS})

add_library(data_class STATIC ${data_class_SRCS})

target_link_libraries(data_class
	${Boost_LIBRARIES}
)

#target_include_directories(data_class PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})