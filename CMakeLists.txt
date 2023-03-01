cmake_minimum_required(VERSION 3.5)
project (Robustness)

find_package(PkgConfig REQUIRED) 
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# IF necessary add paths to PKG_CONFIG_PATH
# export PKG_CONFIG_PATH=$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig:$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/:$PKG_CONFIG_PATH
pkg_check_modules(PETSC_PKG REQUIRED IMPORTED_TARGET PETSc)
pkg_check_modules(SLEPC_PKG REQUIRED IMPORTED_TARGET slepc)


include_directories(eigen)
find_package(OpenMP)
add_subdirectory(networkit)

add_executable(robustness main.cpp cpp/greedy.cpp cpp/laplacian.cpp cpp/dynamicLaplacianSolver.cpp cpp/robustnessGreedy.cpp cpp/robustnessUSTGreedy.cpp cpp/slepc_adapter.cpp)

target_compile_options(robustness PUBLIC -g -O3 -fno-omit-frame-pointer)
target_compile_options(networkit PUBLIC -Wno-deprecated-declarations -g -O3 -fno-omit-frame-pointer)

target_include_directories(robustness PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(robustness INTERFACE )


target_link_libraries(robustness PUBLIC networkit)
target_link_libraries(robustness PUBLIC OpenMP::OpenMP_CXX)
# For address sanitizing link to "asan"
# target_link_libraries(robustness PUBLIC asan)



if(MPI_COMPILE_FLAGS)
  set_target_properties(robustness PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(robustness PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(robustness PUBLIC
        ${MPI_LIBRARIES}
        PkgConfig::PETSC_PKG
        PkgConfig::SLEPC_PKG)



set_target_properties(networkit
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

set_target_properties(robustness
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

install(TARGETS robustness DESTINATION bin)
