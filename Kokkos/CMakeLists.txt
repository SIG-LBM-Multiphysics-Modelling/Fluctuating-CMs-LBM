cmake_minimum_required(VERSION 3.16)

# Project name and version
project(EHD-LBM VERSION 1.0)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Find and link Kokkos
find_package(Kokkos REQUIRED)

# Add the executable by specifying the main source file
# Inconsistent models
add_executable(D2Q9_CMS src/2D/D2Q9_CMS_Fluct_Tests.cpp)
add_executable(D2Q9_BGK src/2D/D2Q9_BGK_Fluct_Tests.cpp)
add_executable(D2Q9_RMS src/2D/D2Q9_RMS_Fluct_Tests.cpp)
add_executable(D2Q9_Reg src/2D/D2Q9_Reg_Fluct_Tests.cpp)
add_executable(D3Q27_CMS src/3D/Q27/D3Q27_CMS_Fluct_Tests.cpp)
add_executable(D3Q27_BGK src/3D/Q27/D3Q27_BGK_Fluct_Tests.cpp)
add_executable(D3Q27_RMS src/3D/Q27/D3Q27_RMS_Fluct_Tests.cpp)
add_executable(TGV src/3D/Q27/D3Q27_CMS_Fluct_Tests_complete_case6.cpp)

# Add the include directory for headers
target_include_directories(D2Q9_CMS PRIVATE include)
target_include_directories(D2Q9_RMS PRIVATE include)
target_include_directories(D2Q9_BGK PRIVATE include)
target_include_directories(D2Q9_Reg PRIVATE include)
target_include_directories(D3Q27_CMS PRIVATE include)
target_include_directories(D3Q27_BGK PRIVATE include)
target_include_directories(D3Q27_RMS PRIVATE include)
target_include_directories(TGV PRIVATE include)

# Link Kokkos and stdc++fs to the target
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.0")
    # For GCC < 9, link against stdc++fs
    target_link_libraries(D2Q9_CMS Kokkos::kokkos stdc++fs)
    target_link_libraries(D2Q9_BGK Kokkos::kokkos stdc++fs)
    target_link_libraries(D2Q9_RMS Kokkos::kokkos stdc++fs)
    target_link_libraries(D2Q9_Reg Kokkos::kokkos stdc++fs)
    target_link_libraries(D3Q27_CMS Kokkos::kokkos stdc++fs)
    target_link_libraries(D3Q27_BGK Kokkos::kokkos stdc++fs)
    target_link_libraries(D3Q27_RMS Kokkos::kokkos stdc++fs)
    target_link_libraries(TGV Kokkos::kokkos stdc++fs)
else()
    # For GCC 9 and later, stdc++fs is not needed
    target_link_libraries(D2Q9_CMS Kokkos::kokkos)
    target_link_libraries(D2Q9_BGK Kokkos::kokkos)
    target_link_libraries(D2Q9_RMS Kokkos::kokkos)
    target_link_libraries(D2Q9_Reg Kokkos::kokkos)
    target_link_libraries(D3Q27_CMS Kokkos::kokkos)
    target_link_libraries(D3Q27_BGK Kokkos::kokkos)
    target_link_libraries(D3Q27_RMS Kokkos::kokkos)
    target_link_libraries(TGV Kokkos::kokkos)
endif()
