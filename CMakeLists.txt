cmake_minimum_required(VERSION 3.14)
project(OpenCompiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Apple Silicon Detection ---
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        message(STATUS "Build: Detected Apple Silicon (arm64) for OpenCompiler")
        set(APPLE_SILICON TRUE)
        set(CMAKE_OSX_ARCHITECTURES arm64)
    else()
        message(STATUS "Build: Detected macOS on Intel processor for OpenCompiler")
        set(APPLE_SILICON FALSE)
    endif()
endif()

# --- MLIR/LLVM Setup (Conceptual for C++ passes) ---
# In a real scenario where C++ files implement MLIR passes, you'd find MLIR and LLVM.
# This requires MLIR/LLVM to be installed in a way CMake can find them (e.g., via CMAKE_PREFIX_PATH).
# For example:
# find_package(MLIR REQUIRED CONFIG)
# find_package(LLVM REQUIRED CONFIG)
# include_directories(${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
# add_definitions(${MLIR_DEFINITIONS} ${LLVM_DEFINITIONS})

# --- C++ Shared Library for Custom Ops/Passes ---
set(CPP_OPS_SRC src/optimizations/arm_vectorizer.cpp)
add_library(opencompiler_cpp_ops SHARED ${CPP_OPS_SRC})
target_compile_features(opencompiler_cpp_ops PRIVATE cxx_std_17)

if(APPLE_SILICON)
    target_compile_definitions(opencompiler_cpp_ops PRIVATE OPENCOMPILER_APPLE_SILICON)
endif()

# Link against MLIR/LLVM libraries if the C++ code actually used their APIs
# target_link_libraries(opencompiler_cpp_ops PRIVATE
#     MLIRIR MLIRPass MLIRDialectFunc # Example MLIR components
#     LLVMCore LLVMSupport # Example LLVM components
# )

# --- Installation of the C++ Library ---
# This installs the shared library into a location where the Python package might find it.
# For instance, it could be placed within the aicompiler Python package directory.
# The exact path depends on how you structure your Python package and native extensions.
# A common approach for Python extensions is to use scikit-build or setuptools with a CMake extension.

# Example: Install to a 'lib' directory within the project's build output for CMake.
# The Python build process (e.g., setup.py using scikit-build) would then pick it up.
install(TARGETS opencompiler_cpp_ops
    LIBRARY DESTINATION lib  # Installs to <prefix>/lib
    RUNTIME DESTINATION bin  # For Windows DLLs, also installs to <prefix>/bin
    ARCHIVE DESTINATION lib  # For static libs or import libs on Windows
)

# --- Executable for testing C++ code directly (Optional) ---
# This can be useful for debugging C++ components independently of Python.
# add_executable(opencompiler_cpp_test src/optimizations/arm_vectorizer.cpp)
# if(TARGET opencompiler_cpp_test AND TARGET opencompiler_cpp_ops)
    # If test executable needs to link the lib we just defined (e.g. to test its functions)
    # target_link_libraries(opencompiler_cpp_test PRIVATE opencompiler_cpp_ops)
# endif()


message(STATUS "CMake configuration finished for OpenCompiler.")
message(STATUS "APPLE_SILICON flag set to: ${APPLE_SILICON}")
message(STATUS "OpenCompiler C++ ops library target: opencompiler_cpp_ops") 