PROJECT(features)
#SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
cmake_policy(SET CMP0015 NEW)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
 

# Libraries
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# To use GDAL, HDF5 libraries
# install GDAL and HDF5 from anaconda
# add the directory contains gdal*.lib, hdf*.lib
link_directories("D:/Anaconda/Library/lib")
# add the directory contains gdal*.hpp, hdf*.hpp
include_directories("D:/Anaconda/Library/include")

file(GLOB HDF_DLLS "D:/Anaconda/Library/bin/hdf*.dll")
file(GLOB HDF_LIBS "D:/Anaconda/Library/lib/hdf*.lib")

# To use MPEG-7 libraries
add_subdirectory(MPEG-7)

 
add_library(feature
src/glcm.cpp
src/glcm.hpp
src/elbp.hpp
src/elbp.cpp
src/cvFeatures.cpp
src/cvFeatures.hpp
src/ober.hpp
src/ober.cpp
src/morph.cpp
src/morph.hpp
src/sarFeatures.hpp
src/sarFeatures.cpp
src/Utils.cpp
src/Utils.h
src/specklefilter.hpp
src/specklefilter.cpp
src/KNN.hpp
src/KNN.cpp
src/cv_hdf5.cpp
src/cv_hdf5.hpp
src/dataset_hdf5.hpp
src/dataset_hdf5.cpp
src/DataProcess.hpp
src/DataProcess.cpp

tsne/sptree.cpp
tsne/sptree.h
tsne/tsne.cpp
tsne/tsne.h
tsne/vptree.h
)

add_executable(main
src/main.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET feature PROPERTY CXX_STANDARD 17)


# libs to link with
target_link_libraries(feature ${OpenCV_LIBS} ${HDF_LIBS}   MPEG-7 )
target_link_libraries(main feature)

# The following code block is suggested to be used on Windows.
 if (MSVC)
     add_custom_command(TARGET main
                     POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${HDF_DLLS}
                      $<TARGET_FILE_DIR:main>)
	add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB -D_HDF5USEDLL_)
	
    
 endif (MSVC)