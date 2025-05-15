"""
JIT execution engine for compiled LLVM modules.

This module provides a JIT compiler and executor for LLVM modules,
with optimizations for Apple Silicon.
"""

import ctypes
import os
import numpy as np
import logging
import sys
import platform

# Add the parent directory to the path to import from compiler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compiler.core import TargetArch

logger = logging.getLogger(__name__)

class JITExecutor:
    """
    JIT execution engine for compiled LLVM modules.
    
    Uses llvmlite to dynamically compile LLVM IR into machine code for immediate execution.
    Handles target-specific configurations for CPU features (e.g., AVX, NEON) and
    provides a basic mechanism for function caching and execution via ctypes.
    """
    
    def __init__(self, target_arch=None):
        if target_arch is None:
            self.target_arch = TargetArch.detect_host()
        else:
            self.target_arch = target_arch
            
        self.llvm = None
        self.target_machine = None
        self.engine = None
        self._init_llvm_engine()
        
        self.modules = {} # Stores {name: llvm_module_ref}
        self.function_cache = {} # Stores {func_name: ctypes_func_ptr}
        
        logger.info(f"Initialized JIT executor for {self.target_arch.name}")
    
    def _init_llvm_engine(self):
        """Initializes the llvmlite JIT engine and target machine."""
        try:
            from llvmlite import binding as llvm
            self.llvm = llvm
            
            llvm.initialize()
            llvm.initialize_native_target()
            llvm.initialize_native_asmprinter()
            
            target = llvm.Target.from_default_triple()
            
            # Configure CPU features based on architecture
            cpu_name = "generic"
            features_str = ""
            if self.target_arch == TargetArch.X86_64_AVX2:
                cpu_name = "haswell" # AVX2 baseline often associated with Haswell
                features_str = "+avx2,+fma"
            elif self.target_arch == TargetArch.X86_64_AVX512:
                cpu_name = "skylake-avx512" # Skylake-X for AVX512
                features_str = "+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"
            elif self.target_arch == TargetArch.APPLE_SILICON:
                # For widespread compatibility, "apple-m1" or a generic arm64 with neon might be safer
                # This depends on the LLVM version linked with llvmlite.
                proc_lower = platform.processor().lower()
                if "m2" in proc_lower:
                    cpu_name = "apple-m2" # Or a more generic target for M2 like 'apple-m1' or specific codename
                elif "m1" in proc_lower:
                    cpu_name = "apple-m1"
                else: # Default for Apple Silicon ARM64 if specific M-chip not identified
                    cpu_name = "apple-a14" # A generic recent Apple ARM core, or more broadly "arm64"
                features_str = "+neon,+fp-armv8,+crypto" # Basic features for Apple Silicon ARMv8
            elif self.target_arch in [TargetArch.ARM64, TargetArch.ARM64_NEON]:
                cpu_name = "generic" # or specific arm core like 'cortex-a72'
                features_str = "+neon,+fp-armv8"
            elif self.target_arch == TargetArch.ARM64_SVE:
                cpu_name = "generic"
                features_str = "+sve,+fp-armv8"
            
            self.target_machine = target.create_target_machine(
                cpu=cpu_name, features=features_str, opt=3 # Use high optimization level for JIT
            )
            
            # Create an empty module to initialize the execution engine
            # Some JIT engines require an initial module to be created with.
            # Here, we provide a minimal, empty module for this purpose.
            initial_module = llvm.parse_assembly("""
                ; ModuleID = 'initial_empty_module'
                source_filename = "initial_empty_module"
            """)
            initial_module.name = "jit_initial_module"
            self.engine = llvm.create_mcjit_compiler(initial_module, self.target_machine)
            logger.info(f"LLVM JIT engine initialized for target: {target.name}, CPU: {cpu_name}, Features: {features_str}")

        except ImportError:
            logger.error("llvmlite is not installed. JITExecutor will not function.")
            self.llvm = None
        except Exception as e:
            logger.error(f"Failed to initialize LLVM JIT engine: {e}")
            self.llvm = None # Ensure it's None if initialization fails
    
    def add_llvm_module(self, llvm_ir_str, name=None):
        if not self.engine or not self.llvm:
            logger.error("LLVM JIT engine not initialized. Cannot add module.")
            return None
            
        if name is None:
            name = f"jit_module_{len(self.modules)}"
            
        try:
            mod = self.llvm.parse_assembly(llvm_ir_str)
            mod.name = name
            mod.verify()

            # Optional: Run LLVM optimization passes on the module before JITting
            # This would require creating a PassManager and populating it.
            # For simplicity in PoC, we rely on the opt level set in TargetMachine.
            # Example:
            # pmb = self.llvm.PassManagerBuilder()
            # pmb.opt_level = 3 # Match JIT engine opt level
            # module_pass_manager = self.llvm.ModulePassManager()
            # pmb.populate(module_pass_manager)
            # module_pass_manager.run(mod)

            self.engine.add_module(mod)
            # Finalize makes the functions callable. Do this after all modules for a logical unit are added,
            # or per module if they are independent.
            # For simplicity, finalizing here. For complex scenarios (e.g. cross-module calls before all are added),
            # finalization might be deferred until all necessary modules are loaded.
            self.engine.finalize_object() 
            
            self.modules[name] = mod # Store the llvmlite module object for potential inspection
            logger.info(f"Added LLVM module '{name}' to JIT executor and finalized.")
            # Clear function cache for this module name if it's being redefined, 
            # as function addresses might change.
            keys_to_remove = [k for k in self.function_cache if k.startswith(f"{name}::")]
            for k in keys_to_remove: del self.function_cache[k]

            return name
        except Exception as e:
            logger.error(f"Failed to add LLVM module '{name}': {e}")
            logger.error(f"LLVM IR causing error (first 500 chars):\n{llvm_ir_str[:500]}")
            return None
    
    def _get_ctypes_func(self, module_name, function_name, arg_types_np, result_type_np):
        """Helper to get a ctypes callable function from JITted code."""
        cache_key = f"{module_name}::{function_name}"
        if cache_key in self.function_cache:
            return self.function_cache[cache_key]

        if not self.engine:
            raise RuntimeError("JIT engine not initialized.")

        try:
            func_ptr = self.engine.get_function_address(function_name)
        except Exception as e: # Could be UnboundGlobalError if function not found
            logger.error(f"Function '{function_name}' not found in JITted module '{module_name}'. Error: {e}")
            # Check available functions for debugging
            # Note: Iterating self.engine.modules might not directly give function names easily.
            # This requires inspecting the LLVM module objects stored in self.modules.
            if module_name in self.modules:
                mod = self.modules[module_name]
                available_funcs = [fn.name for fn in mod.functions if not fn.is_declaration]
                logger.info(f"Available functions in module '{module_name}': {available_funcs}")
            raise NameError(f"Function '{function_name}' not found in JITted module '{module_name}'.") from e

        if not func_ptr:
            raise NameError(f"Function '{function_name}' found but address is null (get_function_address returned 0).")

        # Convert NumPy dtypes to ctypes. This mapping needs to be comprehensive
        # for all data types the JITted functions might handle.
        # The current _np_dtype_to_ctype is a helper method in this class.
        ctypes_arg_types = []
        for arg_np_type in arg_types_np:
            # Assuming all arguments are pointers to the data type
            ctypes_arg_types.append(ctypes.POINTER(self._np_dtype_to_ctype(arg_np_type)))
        
        # Result type also assumed to be a pointer for array outputs
        # If function returns a scalar, this would be different.
        ctypes_result_type = ctypes.POINTER(self._np_dtype_to_ctype(result_type_np)) if result_type_np is not None else None

        cfunc = ctypes.CFUNCTYPE(ctypes_result_type, *ctypes_arg_types)(func_ptr)
        self.function_cache[cache_key] = cfunc
        return cfunc

    def execute(self, function_name, args_np_list, module_name=None):
        """
        Execute a function from a JITted module.
        Args:
            function_name: Name of the function to execute.
            args_np_list: List of NumPy arrays as arguments.
            module_name: Optional name of the module containing the function.
                         If None, tries to find the function in the most recently added module
                         or any module if the name is unique.
                         It's safer to provide the module name if known.
        Returns:
            Function result as a NumPy array (simplified for PoC).
        """
        if not self.engine:
            logger.error("JIT engine not initialized. Cannot execute.")
            # Fallback for PoC: return zeros based on first input shape if any
            return np.zeros(args_np_list[0].shape if args_np_list else (1,), dtype=np.float32)

        if module_name is None:
            # Try to infer module name if function_name is qualified like "mod::func"
            if "::" in function_name:
                module_name, function_name = function_name.split("::", 1)
            else:
                # Search for the function in all modules. This can be ambiguous.
                # A better approach would be to require module_name or have a default/current module context.
                found_in_module = None
                for mod_name, mod_obj in self.modules.items():
                    if any(f.name == function_name for f in mod_obj.functions if not f.is_declaration):
                        if found_in_module is not None:
                            raise NameError(f"Function '{function_name}' is ambiguous; found in multiple modules. Specify module_name.")
                        found_in_module = mod_name
                if found_in_module is None:
                    raise NameError(f"Function '{function_name}' not found in any loaded JIT module.")
                module_name = found_in_module
                logger.info(f"Inferred module '{module_name}' for function '{function_name}'.")

        if not args_np_list:
            raise ValueError("Argument list cannot be empty for execution.")

        # A real implementation would need type information from the LLVM IR/function signature
        # to correctly marshal arguments and prepare for the result.
        arg_types_np = [arg.dtype for arg in args_np_list]
        
        # Simplified: Assume output shape and type needs to be pre-allocated and passed as pointer.
        # This is a common pattern for C functions modifying buffers in place.
        # For now, create a placeholder output based on the first input's shape.
        # THIS IS A MAJOR SIMPLIFICATION and assumes the JITted function works this way.
        output_shape = args_np_list[0].shape 
        output_dtype_np = np.float32 # Assume float32 output for PoC
        output_array = np.zeros(output_shape, dtype=output_dtype_np)

        try:
            # For PoC, assume all inputs are pointers, and output is also a pointer argument (last one).
            # The LLVM function signature must match this expectation.
            # Example C signature: void func(float* in1, ..., float* out_buf)
            effective_arg_types_np = arg_types_np + [output_dtype_np] 
            cfunc = self._get_ctypes_func(module_name, function_name, effective_arg_types_np, result_type_np=None)

            ctypes_args = [arg.ctypes.data_as(ctypes.POINTER(self._np_dtype_to_ctype(arg.dtype))) for arg in args_np_list] 
            ctypes_args.append(output_array.ctypes.data_as(ctypes.POINTER(self._np_dtype_to_ctype(output_array.dtype))))

            logger.debug(f"Executing JITted function '{module_name}::{function_name}' with {len(args_np_list)} inputs.")
            
            # Execute based on target architecture specifics (conceptual for PoC)
            if self.target_arch == TargetArch.APPLE_SILICON:
                self._execute_apple_silicon(cfunc, ctypes_args)
            else:
                cfunc(*ctypes_args)
            
            return output_array # Return the pre-allocated (now filled) output array

        except Exception as e:
            logger.error(f"Error executing JITted function '{module_name}::{function_name}': {e}")
            # Fallback for PoC
            return np.zeros(args_np_list[0].shape if args_np_list else (1,), dtype=np.float32)
    
    def _execute_apple_silicon(self, cfunc, ctypes_args_with_out_ptr):
        """Conceptual special execution path for Apple Silicon for JITted code."""
        # In a real implementation, this might involve hints for thread affinity specific to
        # Apple's Performance/Efficiency cores if exposed, or if the JITted code itself
        # contains calls to Metal/ANE wrappers that were part of the LLVM IR.
        # For this PoC, it's a standard C function call.
        logger.debug("Using Apple Silicon JIT execution path (currently standard call).")
        cfunc(*ctypes_args_with_out_ptr)

    # Helper to map numpy dtype to ctypes, needed by _get_ctypes_func
    def _np_dtype_to_ctype(self, np_dtype):
        """Converts a NumPy dtype to its ctype equivalent."""
        if np_dtype == np.float32: return ctypes.c_float
        if np_dtype == np.float64: return ctypes.c_double
        if np_dtype == np.int32: return ctypes.c_int32
        if np_dtype == np.int64: return ctypes.c_int64
        # Add more mappings as needed for other types (uints, bool, etc.)
        raise NotImplementedError(f"NumPy dtype {np_dtype} to ctype conversion not supported.") 