"""
PyTorch integration for the AI Compiler platform.

This module handles the conversion of PyTorch models to MLIR.
"""

import torch
import torch.fx as fx
import logging
import platform
import sys
import os
import numpy as np

# Add the parent directory to the path to import from compiler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compiler.core import CPUCompiler, TargetArch
from runtime.jit_executor import JITExecutor # For JIT compilation
import mlir.dialects.torch as torch_dialect # For MLIR conversion

logger = logging.getLogger(__name__)

class PyTorchFrontend:
    """
    PyTorch integration for the AI Compiler platform.
    
    This class handles the conversion of PyTorch models to MLIR
    representations that can be processed by the compiler.
    Supports both x86-64 and ARM64 architectures, including Apple Silicon.
    """
    
    def __init__(self, compiler=None):
        """
        Initialize the PyTorch frontend.
        
        Args:
            compiler: CPUCompiler instance (creates a default one if None)
        """
        # Auto-detect best compiler settings if not provided
        if compiler is None:
            # For Apple Silicon, use special optimizations
            if platform.processor().lower().startswith('apple'):
                compiler = CPUCompiler(target_arch=TargetArch.APPLE_SILICON, optimization_level=3)
            else:
                compiler = CPUCompiler()
                
        self.compiler = compiler
        logger.info(f"Initialized PyTorch frontend with compiler target: {compiler.target_arch.name}")
    
    def import_model(self, model, example_inputs=None, use_fx=True, preserve_shape_info=True):
        """
        Import a PyTorch model for compilation.
        
        Args:
            model: PyTorch model to import
            example_inputs: Example inputs for tracing
            use_fx: Whether to use torch.fx for capturing (fallback to JIT)
            preserve_shape_info: Keep shape information for better optimization
            
        Returns:
            MLIR representation of the model as a string
        """
        # Tracing requires example inputs to understand the model structure and data flow.
        # FX tracing is generally preferred for its rich symbolic information, but JIT trace is a fallback.
        if example_inputs is not None:
            is_mps_compatible = False # Assume not compatible initially
            if self.compiler.target_arch == TargetArch.APPLE_SILICON:
                # Attempt to check MPS compatibility by moving model and inputs to MPS device.
                # This is a practical check; more sophisticated static analysis could also be used.
                try:
                    if torch.backends.mps.is_available():
                        mps_device = torch.device("mps")
                        # Ensure example_inputs is a tensor or a tuple/list of tensors
                        if isinstance(example_inputs, torch.Tensor):
                            mps_inputs = example_inputs.to(mps_device)
                        elif isinstance(example_inputs, (list, tuple)):
                            mps_inputs = tuple(inp.to(mps_device) for inp in example_inputs)
                        else:
                            raise TypeError(f"Unsupported example_inputs type: {type(example_inputs)}")
                        
                        model_copy = model.to(mps_device)
                        with torch.no_grad():
                            if isinstance(mps_inputs, tuple):
                                _ = model_copy(*mps_inputs)
                            else:
                                _ = model_copy(mps_inputs)
                        is_mps_compatible = True
                        logger.info("Model is MPS compatible, will use Metal-optimized paths")
                except (AttributeError, RuntimeError, TypeError) as e:
                    logger.warning(f"Model not compatible with MPS or input type error: {e}")
            
            if use_fx and hasattr(fx, "symbolic_trace"):
                try:
                    traced_model = fx.symbolic_trace(model)
                    logger.info("Successfully traced model using torch.fx")
                except Exception as e:
                    logger.warning(f"Failed to trace with fx: {e}, falling back to torch.jit")
                    traced_model = torch.jit.trace(model, example_inputs)
            else:
                traced_model = torch.jit.trace(model, example_inputs)
                
            if self.compiler.target_arch == TargetArch.APPLE_SILICON:
                # Store MPS compatibility on the traced model if possible
                # JIT ScriptModule might not allow arbitrary attribute assignment
                if hasattr(traced_model, '__dict__'):
                    traced_model._mps_compatible = is_mps_compatible 
                else:
                    # For ScriptModule, we might need another way or just rely on context
                    logger.info("Cannot directly annotate ScriptModule with MPS compatibility.")

        else:
            # Models without example inputs can currently only be imported using FX symbolic_trace.
            if not use_fx or not hasattr(fx, "symbolic_trace"):
                raise ValueError("Example inputs are required when not using torch.fx or if fx is unavailable.")
            traced_model = fx.symbolic_trace(model)
        
        return self._convert_to_mlir(traced_model, preserve_shape_info, is_mps_compatible if example_inputs is not None else False)
    
    def _convert_to_mlir(self, traced_model, preserve_shape_info=True, is_mps_compatible=False):
        """Convert traced PyTorch model to MLIR."""
        context = self.compiler.context
        
        # Conditional conversion path based on MPS compatibility for Apple Silicon.
        # This allows for potentially different MLIR generation strategies.
        if self.compiler.target_arch == TargetArch.APPLE_SILICON and is_mps_compatible:
             # Prefer MPS-optimized path if compatible and on Apple Silicon
            with context:
                logger.info("Converting to MLIR using MPS-optimized path.")
                module = self._convert_to_mlir_mps(traced_model, context, preserve_shape_info)
                return str(module)
        
        # Standard conversion path
        with context:
            logger.info("Converting to MLIR using standard torch_dialect importer.")
            # Ensure traced_model is a ScriptModule or compatible type for import_torch_module
            if not isinstance(traced_model, (torch.jit.ScriptModule, torch.fx.GraphModule)):
                # This case should ideally be handled by earlier tracing steps
                logger.warning(f"Traced model is of type {type(traced_model)}, attempting JIT trace.")
                # This might not work if example_inputs were not provided initially
                # This is a fallback, robust handling requires example_inputs for JIT tracing
                try:
                    # We need example inputs here if we are forced to JIT trace now
                    # This indicates a potential logic flaw if we reach here without them
                    # For now, assume it was an FX graph that needs scripting
                    if isinstance(traced_model, torch.fx.GraphModule):
                         traced_model = torch.jit.script(traced_model)
                    else:
                        raise ValueError("Cannot convert non-ScriptModule/GraphModule without example_inputs.")
                except Exception as e:
                    logger.error(f"Fallback JIT tracing failed: {e}")
                    raise

            module = torch_dialect.import_torch_module(
                traced_model, 
                context, 
                preserve_shapes=preserve_shape_info
            )
            return str(module)
    
    def _convert_to_mlir_mps(self, traced_model, context, preserve_shape_info=True):
        """Convert traced PyTorch model to MLIR with Metal optimizations."""
        from mlir.passmanager import PassManager
        
        # The standard torch_dialect importer is used first.
        # Subsequent passes would then lower parts of this to GPU/MPS-specific MLIR constructs.
        module = torch_dialect.import_torch_module(traced_model, context, preserve_shapes=preserve_shape_info)
        
        # Apply Metal-specific transformations
        # The pass pipeline "torch-to-gpu,convert-gpu-to-spirv" is an example.
        # Actual passes for targeting MPS from MLIR's torch dialect would depend on available
        # MLIR capabilities (e.g., `convert-torch-to-linalg`, `convert-linalg-to-gpu`, then custom MPS lowering).
        # This is a complex area and assumes availability of such passes in the MLIR installation.
        try:
            pm = PassManager.parse("torch-to-gpu,convert-gpu-to-spirv", context) # Example, replace with actual MPS passes
            pm.run(module)
            logger.info("Applied MPS-specific transformation passes.")
        except Exception as e:
            logger.warning(f"Failed to apply MPS-specific passes: {e}. Using standard Torch dialect MLIR.")
        
        return module
    
    def compile(self, model, example_inputs=None, output_format="llvm", 
                use_fx=True, preserve_shape_info=True):
        """
        Compile a PyTorch model using the CPU compiler.
        """
        mlir_module_str = self.import_model(
            model, 
            example_inputs,
            use_fx=use_fx,
            preserve_shape_info=preserve_shape_info
        )
        return self.compiler.compile(mlir_module_str, output_format)
    
    def jit_compile(self, model, example_inputs):
        """
        JIT compile a PyTorch model for immediate execution.
        """
        # 1. Compile the PyTorch model to LLVM IR using the established pipeline.
        llvm_ir = self.compile(model, example_inputs, output_format="llvm")
        
        # 2. Initialize the JITExecutor with the correct target architecture.
        jit_executor = JITExecutor(target_arch=self.compiler.target_arch)
        module_name = model.__class__.__name__ # Use model name for the JIT module
        jit_executor.add_llvm_module(llvm_ir, name=module_name)
        
        # 3. Determine the entry function name in the LLVM IR.
        # This typically defaults to "forward" or "main" from the PyTorch to MLIR to LLVM lowering.
        # Standardization or dynamic discovery of this name is important for robustness.
        entry_function_name = "forward" 
        
        def compiled_model_executor(*inputs):
            # 4. Prepare inputs: Convert PyTorch tensors to NumPy arrays for the JIT C ABI.
            numpy_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    numpy_inputs.append(inp.detach().cpu().numpy())
                elif isinstance(inp, np.ndarray):
                    numpy_inputs.append(inp)
                else:
                    # Attempt conversion for other types if necessary, or raise error
                    try:
                        numpy_inputs.append(np.array(inp))
                    except Exception as e:
                        raise TypeError(f"Unsupported input type {type(inp)} for JIT execution: {e}")
            
            # The JITExecutor's execute method needs to handle multiple inputs correctly
            # and the function signature from LLVM IR needs to match.
            # The current JITExecutor.execute is a placeholder.
            # For now, assuming it takes a list of numpy arrays.
            result_array = jit_executor.execute(entry_function_name, numpy_inputs)
            
            # Convert result back to PyTorch tensor
            return torch.tensor(result_array)
        
        return compiled_model_executor 