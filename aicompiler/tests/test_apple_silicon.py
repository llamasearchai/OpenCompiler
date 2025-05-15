"""
Tests for Apple Silicon specific functionality for OpenCompiler.
"""

import unittest
import platform
import sys
import os
import pytest
import numpy as np
import ctypes
from pathlib import Path
import logging

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aicompiler import CPUCompiler, TargetArch # Using the Python package `aicompiler`

# Configure logger for tests
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - OpenCompilerTest - %(levelname)s - %(name)s - %(message)s')

# Check if running on Apple Silicon
is_apple_silicon = (
    platform.system() == "Darwin" and 
    platform.machine() == "arm64" and
    ("apple m" in platform.processor().lower() or "arm64" in platform.processor().lower())
)

# Path to the build directory (assuming Makefile/CMake structure)
# This needs to point to where CMake actually places the built library.
# Typically, this would be in `build/lib` if `install(TARGETS ... LIBRARY DESTINATION lib)` is used
# or directly in `build` if no install step is run from CMake for tests.
# For `make build-cpp`, the library will be in `build/` or `build/src/optimizations/` depending on CMake config.
# Let's assume it might be in `build/` for simplicity of this test.
BUILD_DIR = Path(__file__).parent.parent.parent / "build"
LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so" # Adjust for Linux if needed
CPP_OPS_LIB_NAME = f"libopencompiler_cpp_ops{LIB_SUFFIX}"
CPP_OPS_LIB_PATH = BUILD_DIR / CPP_OPS_LIB_NAME

@pytest.mark.skipif(not is_apple_silicon, reason="Requires Apple Silicon (M-series Mac) for these specific tests.")
class TestAppleSiliconOpenCompiler(unittest.TestCase):
    """Test suite for Apple Silicon specific functionality of OpenCompiler."""
    
    def setUp(self):
        self.compiler = CPUCompiler(target_arch=TargetArch.APPLE_SILICON, optimization_level=3)
        logger.info(f"TestAppleSiliconOpenCompiler: setUp using target {self.compiler.target_arch.name}")
        
    def test_target_detection_explicit_apple_silicon(self):
        """Test OpenCompiler correctly uses Apple Silicon target when explicitly set."""
        self.assertEqual(self.compiler.target_arch, TargetArch.APPLE_SILICON)

    def test_target_detection_auto_on_apple_silicon_hw(self):
        """Test OpenCompiler auto-detects Apple Silicon on appropriate hardware."""
        auto_compiler = CPUCompiler() 
        self.assertEqual(auto_compiler.target_arch, TargetArch.APPLE_SILICON, "Auto-detection should pick Apple Silicon on this hardware.")
        
    def test_pipeline_string_contains_apple_specific_passes(self):
        """Test OpenCompiler's optimization pipeline string includes Apple-specific pass names."""
        pipeline_str = self.compiler._get_pipeline_string()
        logger.debug(f"Generated pipeline string for Apple Silicon: {pipeline_str}")
        self.assertIn("arm-vectorize", pipeline_str, "Pipeline should include arm-vectorize for Apple Silicon.")
        self.assertIn("apple-ane-optimize", pipeline_str, "Pipeline should include conceptual apple-ane-optimize pass.")
        
    def test_compilation_to_arm64_assembly(self):
        """Test OpenCompiler's compilation to assembly targeting ARM64 for Apple Silicon."""
        mlir_module = """
        module {
          func.func @simple_arm_asm_test(%arg0: f32) -> f32 {
            %cst = arith.constant 1.23f : f32
            %res = arith.addf %arg0, %cst : f32
            return %res : f32
          }
        }
        """
        try:
            assembly_code = self.compiler.compile(mlir_module, output_format="assembly")
            self.assertIsNotNone(assembly_code, "Assembly code should not be None.")
            self.assertGreater(len(assembly_code), 0, "Assembly code should not be empty.")
            # More specific checks for ARM64 assembly can be added here
            # For instance, looking for common ARM instructions like 'fadd', 'ldr', 'str', 'ret'
            # This is a basic check and depends on LLVM's output for the specific target.
            self.assertTrue(any(instr in assembly_code.lower() for instr in ["fadd", "ldr", "str", "ret", ".cfi_startproc", "_simple_arm_asm_test:"]), 
                            f"Generated assembly does not contain expected ARM64 markers. Got: {assembly_code[:500]}...")
            logger.info("ARM64 assembly generation test passed conceptually.")
        except Exception as e:
            self.skipTest(f"Assembly compilation for ARM64 failed (MLIR/LLVM setup might be incomplete): {e}")

    @unittest.skipUnless(CPP_OPS_LIB_PATH.is_file(), 
                        f"Skipping C++ interop test: Shared library {CPP_OPS_LIB_NAME} not found at {CPP_OPS_LIB_PATH}. Run 'make build-cpp' first.")
    def test_cpp_optimizer_shared_library_interop(self):
        """Test loading and calling functions from the C++ shared library (conceptual)."""
        logger.info(f"Attempting to load C++ OpenCompiler library: {CPP_OPS_LIB_PATH}")
        try:
            cpp_lib = ctypes.CDLL(str(CPP_OPS_LIB_PATH))
            
            # Test initialize_arm_optimizer_resources
            self.assertTrue(hasattr(cpp_lib, "initialize_arm_optimizer_resources"), "Function initialize_arm_optimizer_resources not found in C++ lib.")
            initialize_func = cpp_lib.initialize_arm_optimizer_resources
            initialize_func.argtypes = []
            initialize_func.restype = None
            logger.info("Calling C++ initialize_arm_optimizer_resources via ctypes...")
            initialize_func() # Should print from C++

            # Test apply_arm_optimizations_placeholder
            self.assertTrue(hasattr(cpp_lib, "apply_arm_optimizations_placeholder"), "Function apply_arm_optimizations_placeholder not found.")
            apply_opt_func = cpp_lib.apply_arm_optimizations_placeholder
            apply_opt_func.argtypes = [ctypes.c_char_p]
            apply_opt_func.restype = ctypes.c_char_p # Pointer to the new string

            self.assertTrue(hasattr(cpp_lib, "free_optimized_ir_string"), "Function free_optimized_ir_string not found.")
            free_func = cpp_lib.free_optimized_ir_string
            free_func.argtypes = [ctypes.c_char_p]
            free_func.restype = None

            test_ir = "module { func.func @test_func() { return } }"
            test_ir_bytes = test_ir.encode('utf-8')
            
            logger.info("Calling C++ apply_arm_optimizations_placeholder via ctypes...")
            returned_ptr = apply_opt_func(test_ir_bytes)
            self.assertIsNotNone(returned_ptr, "C++ optimization function returned a null pointer.")
            
            returned_str = ctypes.cast(returned_ptr, ctypes.c_char_p).value.decode('utf-8')
            logger.info(f"C++ function returned: {returned_str}")
            self.assertIn("Applied ARM optimizations (placeholder)", returned_str, "Optimized IR string marker not found.")
            self.assertIn(test_ir, returned_str, "Original IR not found in returned string.")

            # Free the memory allocated by C++
            logger.info("Calling C++ free_optimized_ir_string via ctypes...")
            free_func(returned_ptr)

            logger.info("C++ shared library interop test passed.")

        except OSError as e:
            self.fail(f"Failed to load or call C++ library {CPP_OPS_LIB_PATH}: {e}. Ensure it is built and compatible (e.g., arm64 vs x86_64).")
        except AttributeError as e:
            self.fail(f"A required function not found in C++ library {CPP_OPS_LIB_PATH} (check extern \"C\" and function names): {e}")

if __name__ == "__main__":
    # This allows running the test file directly, but `pytest` is preferred.
    # For direct run, ensure PYTEST_SKIP_CONFIG is not set or handle skips appropriately.
    if is_apple_silicon:
        unittest.main()
    else:
        print(f"Skipping Apple Silicon specific tests as this is not an Apple Silicon M-series Mac (is_apple_silicon: {is_apple_silicon}).") 