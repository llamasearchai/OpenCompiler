"""
Runtime executor for compiled ML models on CPU.

This module handles loading and executing compiled models on CPU,
with special optimizations for Apple Silicon.
"""

import ctypes
import numpy as np
import logging
from pathlib import Path
import platform
import sys
import os

# Add the parent directory to the path to import from compiler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compiler.core import TargetArch

logger = logging.getLogger(__name__)

class CPURuntimeExecutor:
    """
    Runtime executor for compiled ML models on CPU.
    
    This class handles loading and executing compiled models,
    managing memory, and collecting performance metrics.
    Support for x86-64 and ARM64 architectures, including Apple Silicon.
    """
    
    def __init__(self, num_threads=None, memory_pool_size=None, target_arch=None):
        """
        Initialize the CPU runtime executor.
        
        Args:
            num_threads: Number of threads to use (None = use system default)
            memory_pool_size: Size of memory pool in bytes (None = default)
            target_arch: Target architecture (auto-detected if None)
        """
        self.num_threads = num_threads
        self.memory_pool_size = memory_pool_size
        self.loaded_modules = {}
        
        # Auto-detect target architecture if not provided
        if target_arch is None:
            self.target_arch = TargetArch.detect_host()
        else:
            self.target_arch = target_arch
        
        # Configure thread and memory settings
        self._configure_runtime()
        logger.info(f"Initialized CPU runtime for {self.target_arch.name} with {self.num_threads} threads")
    
    def _configure_runtime(self):
        """Configure runtime thread and memory settings."""
        import os
        if self.num_threads is None:
            # Use all available cores by default
            import multiprocessing
            self.num_threads = multiprocessing.cpu_count()
        
        # Set thread count environment variable
        os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
        
        # Apple Silicon specific configuration
        if self.target_arch == TargetArch.APPLE_SILICON:
            # Enable Metal performance acceleration when available
            try:
                import torch
                if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
                    if torch.backends.mps.is_available():
                        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
                        logger.info("Enabled MPS (Metal Performance Shaders) acceleration for PyTorch (if used).")
            except ImportError:
                pass
                
            # Configure ANE if available
            try:
                # This is a placeholder - actual ANE configuration would depend on 
                # future Apple Neural Engine APIs
                os.environ["ENABLE_ANE"] = "1"
                logger.info("Configured Apple Neural Engine acceleration")
            except Exception as e:
                logger.warning(f"Failed to configure ANE: {e}")
    
    def load_module(self, module_path, name=None):
        """
        Load a compiled module for execution.
        
        Args:
            module_path: Path to compiled module (.so file)
            name: Optional name for referencing this module
            
        Returns:
            Module ID for later reference
        """
        module_path = Path(module_path)
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        module_id = name or module_path.stem
        
        # Platform-specific library loading approach
        if platform.system() == 'Darwin':
            # macOS (including Apple Silicon) shared library loading
            try:
                # Try to load with RTLD_GLOBAL to ensure all symbols are available
                lib = ctypes.CDLL(str(module_path), ctypes.RTLD_GLOBAL)
            except OSError as e:
                logger.error(f"Failed to load library: {e}")
                # Special handling for Apple Silicon - check architecture
                if self.target_arch == TargetArch.APPLE_SILICON:
                    # Check if the library is compatible with arm64
                    import subprocess
                    result = subprocess.run(['file', str(module_path)], capture_output=True, text=True)
                    if 'arm64' not in result.stdout:
                        raise RuntimeError(f"Library {module_path} is not compiled for arm64 architecture")
                raise
        else:
            # Standard library loading for other platforms
            lib = ctypes.CDLL(str(module_path))
        
        # Extract the function handles and metadata
        self.loaded_modules[module_id] = {
            "library": lib,
            "path": module_path,
            "functions": self._extract_functions(lib, module_id)
        }
        
        logger.info(f"Loaded module '{module_id}' from {module_path}")
        return module_id
    
    def _extract_functions(self, lib, module_id):
        """Extract exported functions from the library using ctypes."""
        functions = {}
        # This is a simplified approach for a proof-of-concept.
        # A robust solution would involve:
        # 1. A manifest file or metadata embedded in the shared library describing exported functions,
        #    their names, and their full C signatures (argument types, return type).
        # 2. Standardized naming conventions for functions if a manifest is not used.
        # 3. Potentially using tools like `nm` (platform-dependent) to list symbols, though this
        #    doesn't give signature information directly.

        # We try a list of common or derivable function names.
        # For a model named 'my_model', we might look for 'my_model_forward', 'forward', 'main'.
        func_names_to_try = ["forward", "main", module_id, f"{module_id}_forward", f"run_{module_id}"]
        if module_id.startswith("lib"): # Common prefix for libraries
            func_names_to_try.append(module_id[3:]) # e.g. if module_id is 'libmodel.so' -> try 'model'
        
        for func_name in func_names_to_try:
            try:
                c_func = getattr(lib, func_name)
                # Define a generic signature: (ptr_input_array, ptr_output_array, input_size_elements)
                # This is HIGHLY SIMPLIFIED and likely incorrect for most real-world models.
                # It assumes a single flat input array and a single flat output array of floats,
                # with the total number of input elements passed as the third argument.
                # Real models have multiple inputs/outputs, varying data types, and complex shapes.
                c_func.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # input data buffer
                    ctypes.POINTER(ctypes.c_float),  # output data buffer
                    ctypes.c_int                     # N_elements (e.g., total elements in input_array)
                    # A real signature would need to convey shapes, ranks, multiple tensors, etc.
                    # This might be done via more pointer arguments to structs cottura_code.c_int(input_size_elements))
                ]
                c_func.restype = None # Assuming void return, or int for status code
                functions[func_name] = c_func
                logger.info(f"Found function '{func_name}' in {module_id} with assumed signature.")
                
                # For Apple Silicon, one might look for suffixed versions like 'forward_metal'
                if self.target_arch == TargetArch.APPLE_SILICON:
                    for suffix in ["_metal", "_ane"]:
                        try:
                            accel_func_name = f"{func_name}{suffix}"
                            c_accel_func = getattr(lib, accel_func_name)
                            c_accel_func.argtypes = c_func.argtypes # Assume same sig for now
                            c_accel_func.restype = c_func.restype
                            functions[accel_func_name] = c_accel_func
                            logger.info(f"Found accelerated function '{accel_func_name}'.")
                        except AttributeError:
                            pass # Accelerated version not found
                break # Found a base function, stop trying other names for the base
            except AttributeError:
                continue # Function not found with this name
            except Exception as e:
                logger.warning(f"Error setting up function {func_name} from {module_id}: {e}")

        if not functions:
            logger.warning(f"No standard-named ctypes functions found in {module_id}. Execution will rely on placeholder if defined.")
            # Define a Python callable placeholder if no ctypes function was found
            def placeholder_forward(inputs_list_np, output_shape_tuple):
                logger.warning(f"Executing PYTHON placeholder_forward for {module_id} (no ctypes function loaded). This is for testing structure only.")
                return np.zeros(output_shape_tuple, dtype=np.float32)
            functions["forward"] = placeholder_forward
        return functions
    
    def _execute_ctypes_function(self, module_id, c_func_ptr, inputs_list_np, output_shape_tuple):
        """Helper to execute a ctypes function pointer with a simplified signature."""
        if not isinstance(inputs_list_np, list) or not inputs_list_np:
            raise ValueError("inputs_list_np must be a non-empty list of NumPy arrays.")
        
        # Simplified: assumes first input is the primary data, others might be ignored or handled by C code
        input_array = np.ascontiguousarray(inputs_list_np[0], dtype=np.float32)
        output_array = np.zeros(output_shape_tuple, dtype=np.float32)
        
        input_ptr = input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # The assumed signature has total number of elements for the third arg.
        # This is a major simplification.
        input_size_elements = input_array.size
        
        func_display_name = getattr(c_func_ptr, '_name_', 'ctypes_function')
        logger.debug(f"Executing ctypes function '{func_display_name}' from {module_id} with input_elements={input_size_elements}")
        
        # Execute the C function
        # If restype is int, it could be a status code.
        status = c_func_ptr(input_ptr, output_ptr, ctypes.c_int(input_size_elements))
        
        if c_func_ptr.restype == ctypes.c_int and status != 0:
             logger.error(f"Execution of '{func_display_name}' returned error status: {status}")
             raise RuntimeError(f"Execution of {func_display_name} failed with status {status}")
            
        return output_array

    def execute(self, module_id, function_name, inputs, output_shape):
        if module_id not in self.loaded_modules:
            raise ValueError(f"Module not loaded: {module_id}")
            
        module_data = self.loaded_modules[module_id]
        
        # Prepare inputs as a list of numpy arrays
        numpy_inputs = []
        if not isinstance(inputs, list):
            inputs = [inputs]
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                numpy_inputs.append(np.ascontiguousarray(inp, dtype=np.float32))
            else:
                numpy_inputs.append(np.ascontiguousarray(np.array(inp), dtype=np.float32))

        # Attempt to use accelerated versions on Apple Silicon first
        if self.target_arch == TargetArch.APPLE_SILICON:
            for suffix in ["_ane", "_metal"]:
                accel_func_name = f"{function_name}{suffix}"
                if accel_func_name in module_data["functions"]:
                    func_candidate = module_data["functions"][accel_func_name]
                    if isinstance(func_candidate, ctypes._CFuncPtr): # Check if it's a C function pointer
                        logger.info(f"Using Apple Silicon accelerated ctypes path: {accel_func_name}")
                        try:
                            return self._execute_ctypes_function(module_id, func_candidate, numpy_inputs, output_shape)
                        except Exception as e:
                            logger.warning(f"Execution of ctypes {accel_func_name} failed: {e}. Falling back.")
                    elif callable(func_candidate): # Check if it's a Python placeholder
                        logger.info(f"Using Apple Silicon Python placeholder: {accel_func_name}")
                        return func_candidate(numpy_inputs, output_shape)

        # Fallback to standard function name
        if function_name in module_data["functions"]:
            func_candidate = module_data["functions"][function_name]
            if isinstance(func_candidate, ctypes._CFuncPtr): # C function pointer
                 logger.info(f"Using standard ctypes path: {function_name}")
                 return self._execute_ctypes_function(module_id, func_candidate, numpy_inputs, output_shape)
            elif callable(func_candidate): # Python placeholder (from _extract_functions fallback)
                 logger.info(f"Using Python placeholder: {function_name}")
                 return func_candidate(numpy_inputs, output_shape)
        
        raise ValueError(f"Function '{function_name}' (or its variants/placeholders) not found or not callable in module {module_id}. Available: {list(module_data['functions'].keys())}")
    
    def benchmark(self, module_id, function_name, inputs, output_shape, num_iterations=100):
        import time
        
        logger.info(f"Benchmarking {function_name} from {module_id} for {num_iterations} iterations.")
        # Warmup: essential for JIT-compiled code or to get CPU caches/frequency scaling in a stable state.
        warmup_iterations = min(5, num_iterations // 10 + 1) # At least 1, at most 5 (or 10% of iter)
        logger.debug(f"Running {warmup_iterations} warmup iterations...")
        for _ in range(warmup_iterations):
            self.execute(module_id, function_name, inputs, output_shape)
        
        timings = []
        logger.debug(f"Running {num_iterations} benchmark iterations...")
        for i in range(num_iterations):
            start_time = time.perf_counter()
            self.execute(module_id, function_name, inputs, output_shape)
            end_time = time.perf_counter()
            timings.append(end_time - start_time)
            if (i + 1) % (num_iterations // 10 or 1) == 0:
                 logger.debug(f"Benchmark progress: {i+1}/{num_iterations}")
        
        total_time = sum(timings)
        avg_time_sec = total_time / num_iterations if num_iterations > 0 else 0
        throughput = num_iterations / total_time if total_time > 0 else float('inf')
        
        memory_usage = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            logger.warning("psutil not installed, cannot report memory usage.")
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
        
        results = {
            "inference_time_ms_avg": avg_time_sec * 1000,
            "throughput_inf_per_sec": throughput,
            "memory_usage_mb": memory_usage,
            "num_iterations": num_iterations,
            "total_time_sec": total_time,
            "target_arch": self.target_arch.name,
            "timings_ms_individual": [t * 1000 for t in timings]
        }
        logger.info(f"Benchmark results for {function_name}: {results['inference_time_ms_avg']:.3f} ms/inf, {results['throughput_inf_per_sec']:.2f} inf/sec")
        return results 