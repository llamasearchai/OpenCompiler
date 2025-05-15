#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch example demonstrating the OpenCompiler platform.

This example shows how to:
1. Define a simple PyTorch model.
2. Use the PyTorch frontend (from `aicompiler` package) to compile the model.
3. Execute the model using the JIT runtime (conceptual execution).
"""

import os
import sys
import logging
import torch
import numpy as np
import time

# Add the parent directory to the path to import aicompiler package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aicompiler import PyTorchFrontend, TargetArch # Python package remains aicompiler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - OpenCompiler - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)

class SimpleModel(torch.nn.Module):
    """A simple PyTorch model with a single linear layer."""
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

def main():
    logger.info("--- OpenCompiler PyTorch Example --- ")
    input_size, output_size, batch_size = 10, 5, 1
    model = SimpleModel(input_size, output_size)
    logger.info(f"Created PyTorch model: {model}")
    
    example_inputs = torch.randn(batch_size, input_size)
    logger.info(f"Example inputs shape: {example_inputs.shape}")
    
    with torch.no_grad():
        reference_output = model(example_inputs)
    logger.info(f"Reference PyTorch output shape: {reference_output.shape}")
    
    logger.info("Initializing PyTorchFrontend for OpenCompiler...")
    frontend = PyTorchFrontend() # CPUCompiler is auto-created with host detection
    logger.info(f"OpenCompiler (via PyTorchFrontend) targeting: {frontend.compiler.target_arch.name}")
    
    logger.info("JIT compiling the model with OpenCompiler...")
    # Note: The current JITExecutor and MLIR conversion are placeholders.
    # This will run, but the `compiled_fn` might not be fully optimized or might fallback.
    try:
        compiled_fn = frontend.jit_compile(model, example_inputs)
        logger.info("JIT compilation step completed.")
    except Exception as e:
        logger.error(f"JIT compilation failed: {e}")
        logger.error("This could be due to LLVM/MLIR setup or issues in the JIT path.")
        return

    logger.info("Executing the JIT-compiled model (conceptual)...")
    try:
        result = compiled_fn(example_inputs)
        logger.info(f"OpenCompiler JIT execution result shape: {result.shape}")

        # Comparison will likely fail or be misleading with current placeholder JIT.
        if torch.allclose(result, reference_output, rtol=1e-3, atol=1e-3):
            logger.info("SUCCESS: OpenCompiler JIT result matches PyTorch reference!")
        else:
            logger.warning("OpenCompiler JIT result does NOT exactly match PyTorch reference.")
            logger.warning("This is expected if JIT path or MLIR conversion is not fully functional.")
            logger.debug(f"JIT Result:\n{result}")
            logger.debug(f"Ref Result:\n{reference_output}")

    except Exception as e:
        logger.error(f"JIT execution failed: {e}")
    
    logger.info("Benchmarking execution speed (PyTorch vs Conceptual JIT)...")
    n_iterations = 100
    # Warm up
    for _ in range(10): _ = model(example_inputs); _ = compiled_fn(example_inputs)
    
    torch_times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        with torch.no_grad(): _ = model(example_inputs)
        torch_times.append(time.perf_counter() - t_start)
    torch_avg_time_ms = (sum(torch_times) / n_iterations) * 1000
    logger.info(f"PyTorch native execution: {torch_avg_time_ms:.3f} ms/iteration")

    compiled_times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        _ = compiled_fn(example_inputs)
        compiled_times.append(time.perf_counter() - t_start)
    compiled_avg_time_ms = (sum(compiled_times) / n_iterations) * 1000
    logger.info(f"OpenCompiler JIT execution: {compiled_avg_time_ms:.3f} ms/iteration")

    if compiled_avg_time_ms > 0 and torch_avg_time_ms > 0:
        speedup = torch_avg_time_ms / compiled_avg_time_ms
        logger.info(f"Conceptual speedup: {speedup:.2f}x (May not be meaningful with placeholder JIT)")
    
    if frontend.compiler.target_arch == TargetArch.APPLE_SILICON:
        logger.info("\nOpenCompiler on Apple Silicon: NEON vectorization is a key optimization path. MPS/ANE integration are conceptual.")
    
    logger.info("--- OpenCompiler PyTorch Example Finished --- ")

if __name__ == "__main__":
    main() 