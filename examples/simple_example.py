#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple example demonstrating the OpenCompiler platform.

This example shows how to:
1. Create a compiler with auto-detected architecture using the `aicompiler` package.
2. Compile a simple MLIR module.
3. Print information about the compilation process.
"""

import os
import sys
import logging
import platform

# Add the parent directory to the path to import aicompiler package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aicompiler import CPUCompiler, TargetArch # Python package remains aicompiler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - OpenCompiler - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__) # Use file name for logger

def main():
    logger.info("--- OpenCompiler Simple Example --- ")
    # Print system information
    logger.info(f"System: {platform.system()}, Machine: {platform.machine()}, Processor: {platform.processor()}")
    
    # Create a compiler with auto-detected architecture
    compiler = CPUCompiler()
    logger.info(f"OpenCompiler detected target architecture: {compiler.target_arch.name}")
    logger.info(f"Optimization level: O{compiler.optimization_level}")
    
    # Define a simple MLIR module
    mlir_module = """
    module {
      func.func @simple_add(%arg0: f32, %arg1: f32) -> f32 {
        %0 = arith.addf %arg0, %arg1 : f32
        return %0 : f32
      }
    }
    """
    logger.info(f"Test MLIR module:\n{mlir_module}")
    
    # Try to compile
    try:
        logger.info("Compiling MLIR module to LLVM IR with OpenCompiler...")
        llvm_ir = compiler.compile(mlir_module, output_format="llvm")
        logger.info(f"Compilation successful. LLVM IR size: {len(llvm_ir)} characters.")
        
        lines = llvm_ir.strip().split('\n')[:10]
        logger.info("First 10 lines of generated LLVM IR:")
        for i, line in enumerate(lines):
            logger.info(f"  {i+1:2d}: {line}")
            
    except Exception as e:
        logger.error(f"OpenCompiler compilation failed: {e}")
        logger.error("This might be expected if MLIR/LLVM dependencies are not fully configured.")
        logger.info("The example still demonstrates architecture detection and basic structure.")
        
    pipeline = compiler._get_pipeline_string()
    logger.info(f"OpenCompiler optimization pipeline string: {pipeline}")
    
    if compiler.target_arch == TargetArch.APPLE_SILICON:
        logger.info("Apple Silicon specific considerations for OpenCompiler:")
        logger.info("  - ARM NEON vectorization is prioritized.")
        logger.info("  - Conceptual paths for Metal Performance Shaders (MPS) and Apple Neural Engine (ANE) exist.")
    
    logger.info("--- OpenCompiler Simple Example Finished --- ")

if __name__ == "__main__":
    main() 