PROJECT(features)
#SET(CMAKE_BUILD_TYPE Release)
CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
cmake_policy(SET CMP0015 NEW)
SET(CMAKE_GENERATOR_PLATFORM x64)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# Libraries
#set(OpenCV_FOUND 1)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


#find_package(Torch REQUIRED)
#include_directories(${Torch_INCLUDE_DIRS})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")

# To use GDAL, HDF5 libraries
# install GDAL and HDF5 from anaconda
link_directories(C:/Users/anura/Anaconda3/Library/lib)
include_directories(C:/Users/anura/Anaconda3/Library/include)
set(CMAKE_MSVCIDE_RUN_PATH  C:/Users/anura/Anaconda3/Library/bin)
add_definitions(-DH5_BUILT_AS_DYNAMIC_LIB -D_HDF5USEDLL_)


# To use MPEG-7 libraries
add_subdirectory(src/MPEG-7)

add_library(feature
src/glcm.cpp
src/glcm.hpp
src/elbp.hpp
src/elbp.cpp
src/sen12ms.cpp
src/sen12ms.hpp
src/Geotiff.hpp
src/cvFeatures.cpp
src/cvFeatures.hpp
src/ober.hpp
src/ober.cpp
src/mp.cpp
src/mp.hpp
src/sarFeatures.hpp
src/sarFeatures.cpp
src/Utils.cpp
src/Utils.h
#src/torchDataSet.cpp
src/specklefilter.hpp
src/specklefilter.cpp
src/KNN.hpp
src/KNN.cpp
src/cv_hdf5.cpp
src/cv_hdf5.hpp
)

add_executable(main
src/main.cpp
)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET feature PROPERTY CXX_STANDARD 17)


# libs to link with
target_link_libraries(feature ${OpenCV_LIBS} hdf5 gdal_i MPEG-7)
target_link_libraries(main feature)


# The following code block is suggested to be used on Windows.
#if (MSVC)
#  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#  add_custom_command(TARGET main
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:main>)
#endif (MSVC)
 