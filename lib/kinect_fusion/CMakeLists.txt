project("DynamicFusion")
cmake_minimum_required(VERSION 2.8)

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE Release)
endif()

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# supply compile-time directory to code
set(compileDirectory \"${PROJECT_SOURCE_DIR}\")
add_definitions(-DCOMPILE_DIR=${compileDirectory})

option(BUILD_POLY3 "build for poly3 camera model" on)
if(BUILD_POLY3)
  add_definitions(-DBUILD_POLY3)
endif()

option(BUILD_LINEAR "build for linear camera model" on)
if (BUILD_LINEAR)
  add_definitions(-DBUILD_LINEAR)
endif()

option(BUILD_DOUBLE "build for double precision" off)
if (BUILD_DOUBLE)
  add_definitions(-DBUILD_DOUBLE)
endif()

find_package(CUDA REQUIRED)
find_package(Pangolin REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(PkgConfig REQUIRED)

#pkg_check_modules(nanoflann nanoflann REQUIRED)

# TODO:
set(SUITESPARSE_INCLUDE_DIRS "/usr/include/suitesparse" CACHE PATH "suitesparse include directory")
set(SUITESPARSE_LIBRARIES "cholmod;metis")
#set(EIGEN_DIR "/home/yuxiang/tensorflow/local/lib/python2.7/site-packages/tensorflow/include")
set(EIGEN3_INCLUDE_DIR "D:/SL/eigen-git-mirror")
set(nanoflann_INCLUDE_DIRS "D:/SL/nanoflann/include")
set(Sophus_INCLUDE_DIRS "D:/SL/Sophus-1.0.0")
#set (CMAKE_CXX_STANDARD 11)  -> not supported. Link: https://stackoverflow.com/a/31010221
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  #  Link: https://github.com/jozefizso/protobuf/commit/9eebeb5e4173f235d5f1cd6b001ae5378bd01e4e

# suitesparse libraries /usr/lib/x86_64-linux-gnu/libspqr.so;/usr/lib/x86_64-linux-gnu/libcholmod.so;/usr/lib/x86_64-linux-gnu/libccolamd.so;/usr/lib/x86_64-linux-gnu/libcamd.so;/usr/lib/x86_64-linux-gnu/libcolamd.so;/usr/lib/x86_64-linux-gnu/libamd.so;/usr/lib/liblapack.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so;/usr/lib/libf77blas.so;/usr/lib/libatlas.so;/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so;/usr/lib/x86_64-linux-gnu/librt.so;/usr/lib/x86_64-linux-gnu/libmetis.so


include_directories(${PROJECT_SOURCE_DIR}/include
                    ${Pangolin_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${SUITESPARSE_INCLUDE_DIRS}
                    ${nanoflann_INCLUDE_DIRS}
					${Sophus_INCLUDE_DIRS})

link_directories(${Pangolin_LIBRARY_DIRS})

link_libraries(${Pangolin_LIBRARIES}
               ${SUITESPARSE_LIBRARIES} "-lutil")

file( GLOB DF_HEADERS
  "./include/df/*/*.h"
)

file( GLOB DF_SOURCES
  "./src/*/*.cpp"
)

file( GLOB DF_CUDA_SOURCES
  "./src/*/*.cu"
)

file( GLOB SHADER_SOURCES
  "./shaders/*.vert"
  "./shaders/*.frag"
)

#set(CUDA_SEPARABLE_COMPILATION ON)

if (MSVC)
	message("Set MSVC flags")
	# disable multicore-compilation in MSVS with /MP1. See issue: https://github.com/Xtra-Computing/thundersvm/issues/40  and msvs flag: https://gitlab.kitware.com/cmake/cmake/issues/16318
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /FS /MP1")   # use CMAKE_CXX_FLAGS Link: https://stackoverflow.com/questions/45955272/modern-way-to-set-compiler-flags-in-cross-platform-cmake-project?noredirect=1&lq=1
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr;-arch=sm_61;--expt-extended-lambda;--verbose;-Xcompiler ")  # -fPIC; --std=c++11
else ()
	message("Set gcc flags")
	#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;-Xcompiler -fPIC;")
endif ()
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-gencode arch=compute_61,code=sm_61;--expt-extended-lambda;--verbose;")

message(STATUS "pangolin: ${Pangolin_INCLUDE_DIRS}")

message("DF_HEADERS = ${DF_HEADERS}")

cuda_add_library(
  kfusion
  SHARED
  kinect_fusion.cpp
  ${DF_HEADERS}
  ${DF_SOURCES}
  ${DF_CUDA_SOURCES}
  ${SHADER_SOURCES}
)

cuda_add_executable(
  kinectFusion
  kinect_fusion.cpp
  ${DF_HEADERS}
  ${DF_SOURCES}
  ${DF_CUDA_SOURCES}
  ${SHADER_SOURCES}
)
