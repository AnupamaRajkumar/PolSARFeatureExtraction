cmake_minimum_required(VERSION 3.1)

project( thesis LANGUAGES CXX )

find_package( OpenCV REQUIRED )


add_library(code 
    main.cpp
    Data.cpp
    Data.h
    Feature.cpp
    Feature.h
	KNN.cpp
	KNN.h
	Utils.cpp
	Utils.h
	Performance.cpp
	Performance.h
)

set_target_properties(code PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(code 
    PUBLIC
        ${OpenCV_LIBS}
)


add_executable(main 
    main.cpp 
)

set_target_properties(main PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_link_libraries(main 
    PRIVATE
        code
)

