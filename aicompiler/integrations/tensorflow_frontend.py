import logging
from typing import Optional, Any, Tuple, List, Dict

# Conditional import for TensorFlow
try:
    import tensorflow as tf
    # Ensure we are using TF2 behavior
    if not tf.__version__.startswith("2."):
        logging.warning(f"TensorFlow version {tf.__version__} is not 2.x. Compatibility issues may arise.")
    tf_imported = True
except ImportError:
    logging.info("TensorFlow not found. TensorFlowFrontend will not be available.")
    tf_imported = False

from ..compiler.core import CPUCompiler, TargetArch, CompilationResult
from ..optimizations.graph_optimizer import TensorFlowOptimizer, GraphOptimizer
from ..runtime.jit_executor import JITExecutor # For conceptual JIT

logger = logging.getLogger(__name__)

class TensorFlowFrontend:
    """
    TensorFlow Keras integration for the OpenCompiler platform.
    Handles conversion of TensorFlow models to MLIR and their compilation.
    """
    def __init__(self, compiler: Optional[CPUCompiler] = None, graph_optimizer: Optional[GraphOptimizer] = None):
        if not tf_imported:
            raise ImportError("TensorFlow is not installed. Please install TensorFlow to use TensorFlowFrontend.")
        
        self.compiler = compiler if compiler else CPUCompiler() # Auto-detect host if not specified
        self.graph_optimizer = graph_optimizer if graph_optimizer else TensorFlowOptimizer()
        logger.info(f"Initialized TensorFlowFrontend with compiler target: {self.compiler.target_arch.name}")

    def _get_concrete_function(self, model: tf.keras.Model, example_inputs: Optional[Any] = None) -> Any:
        """Gets a concrete function from a Keras model or a tf.function."""
        if hasattr(model, '__call__') and hasattr(model.call, 'get_concrete_function'): # Keras model
            if example_inputs is not None:
                if isinstance(example_inputs, tuple):
                    return model.call.get_concrete_function(*example_inputs)
                elif isinstance(example_inputs, dict):
                    return model.call.get_concrete_function(**example_inputs)
                else: # Single tensor input
                    return model.call.get_concrete_function(example_inputs)
            else:
                # Attempt to get concrete function without inputs (might require model to be built)
                try:
                    return model.call.get_concrete_function()
                except Exception as e:
                    logger.error(f"Could not get concrete function without example_inputs for Keras model. Model might need to be built or called once. Error: {e}")
                    raise
        elif callable(model) and hasattr(model, 'get_concrete_function'): # tf.function
            if example_inputs is not None:
                if isinstance(example_inputs, tuple):
                    return model.get_concrete_function(*example_inputs)
                elif isinstance(example_inputs, dict):
                    return model.get_concrete_function(**example_inputs)
                else:
                    return model.get_concrete_function(example_inputs)
            else:
                # Attempt to get concrete function without inputs for tf.function
                try:
                    return model.get_concrete_function()
                except Exception as e:
                    logger.error(f"Could not get concrete function without example_inputs for tf.function. Error: {e}")
                    raise
        else:
            raise ValueError("Input 'model' must be a TensorFlow Keras model or a tf.function.")

    def import_model(self, model: tf.keras.Model, example_inputs: Optional[Any] = None, 
                     optimize_graph: bool = True) -> str:
        """
        Import a TensorFlow Keras model or tf.function for compilation.
        Converts the model to MLIR (TensorFlow dialect initially).

        Args:
            model: TensorFlow Keras model or a tf.function.
            example_inputs: Example inputs for tracing to get a concrete function.
                            Can be a single tensor, a tuple of tensors, or a dict.
            optimize_graph: Whether to apply TensorFlow graph optimizations (e.g., Grappler) before MLIR conversion.

        Returns:
            MLIR module string.
        """
        logger.info(f"Importing TensorFlow model: {model.name if hasattr(model, 'name') else 'tf_function'}")
        
        concrete_func = self._get_concrete_function(model, example_inputs)

        # Apply TensorFlow graph optimizations (e.g., Grappler) if requested
        # This happens before MLIR conversion.
        # The MLIR bridge itself might also invoke Grappler.
        # For explicit control, one might use tf.config.optimizer.set_experimental_options here.
        if optimize_graph:
            logger.debug("Applying TensorFlow graph optimizations (Grappler might be invoked by MLIR bridge)...")
            # No explicit API call here, as tf.mlir.experimental.convert_function often does this.
            # If more control is needed, one would use tf.function with grappler options or tf.config.

        # Convert the concrete function to MLIR (TensorFlow dialect)
        try:
            # converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
            # tflite_model = converter.convert()
            # This gives TFLite dialect. We want TF dialect first for CPUCompiler pipeline.
            # Or, use the MLIR bridge directly for TF dialect:
            mlir_module_tf_dialect = tf.mlir.experimental.convert_function(
                concrete_func,
                pass_pipeline='tf-standard-pipeline' # Applies standard TF dialect transformations
            )
            logger.info("TensorFlow model successfully converted to MLIR (TF Dialect).")
        except Exception as e:
            logger.error(f"Failed to convert TensorFlow function to MLIR: {e}")
            raise

        # Conceptual: Apply GraphOptimizer from this project (operates on a generic graph representation or MLIR string)
        # For now, our TensorFlowOptimizer is a placeholder. A real one would parse and transform the MLIR.
        # optimized_mlir_string = self.graph_optimizer.optimize(str(mlir_module_tf_dialect))
        
        return str(mlir_module_tf_dialect)

    def compile(self, model: tf.keras.Model, example_inputs: Optional[Any] = None,
                output_format: str = "llvm", optimize_graph: bool = True) -> CompilationResult:
        """
        Compile a TensorFlow Keras model or tf.function using the OpenCompiler.

        Args:
            model: TensorFlow Keras model or tf.function.
            example_inputs: Example inputs for tracing.
            output_format: Target output format ("llvm", "object", "assembly").
            optimize_graph: Whether to apply TensorFlow graph optimizations before MLIR conversion.

        Returns:
            CompilationResult containing the compiled artifact.
        """
        logger.info(f"Compiling TensorFlow model '{model.name if hasattr(model, 'name') else 'tf_function'}' to {output_format} for target {self.compiler.target_arch.name}")
        mlir_tf_dialect_str = self.import_model(model, example_inputs, optimize_graph)
        
        # The CPUCompiler will take this TF dialect MLIR and run its pipeline
        # (TF -> Linalg -> SCF -> Vector -> LLVM)
        return self.compiler.compile(mlir_tf_dialect_str, output_format)

    def jit_compile(self, model: tf.keras.Model, example_inputs: Optional[Any] = None,
                    optimize_graph: bool = True) -> Callable[..., Any]:
        """
        JIT compile a TensorFlow model for immediate execution (conceptual).

        Args:
            model: TensorFlow Keras model or tf.function.
            example_inputs: Example inputs for tracing.
            optimize_graph: Apply TF graph optimizations.

        Returns:
            An executable function that can run the model.
        """
        if not tf_imported:
            raise ImportError("TensorFlow is not installed. Cannot JIT compile.")

        logger.info(f"JIT compiling TensorFlow model '{model.name if hasattr(model, 'name') else 'tf_function'}' for target {self.compiler.target_arch.name}")        
        compilation_result = self.compile(model, example_inputs, output_format="llvm", optimize_graph=optimize_graph)
        llvm_ir = compilation_result.artifact

        # Create JIT executor with appropriate engine for the target architecture
        jit_executor = JITExecutor(target_arch=self.compiler.target_arch)
        module_id = jit_executor.add_llvm_module(llvm_ir) # Name is auto-generated
        
        # Need to determine the actual function name within the LLVM IR.
        # tf.mlir.experimental.convert_function typically wraps the main function.
        # It often becomes something like "main" or derived from the concrete function name.
        # This requires inspecting the LLVM IR or knowing the naming convention.
        # For now, let's assume a common name like "main" or a predictable one.
        # A robust way is to parse the LLVM IR for public function names.
        # We also need to know the function signature for ctypes.
        
        # Placeholder: Extract function name and signature (this is highly complex)
        # For this example, let's assume the primary function is named `main`
        # and it takes arguments matching `example_inputs` and returns outputs.
        # The `JITExecutor.execute` method will need to handle type conversions.

        # Try to find the main function name in the LLVM IR (simplistic search)
        func_name_to_call = "main" # Default guess
        # A better way would be to inspect module.functions from llvmlite after parsing

        logger.warning("JIT execution for TensorFlow is conceptual. Function name and signature discovery is complex.")
        logger.warning(f"Assuming JIT function name '{func_name_to_call}'. This might require adjustment.")

        def compiled_model_executor(*inputs: Any) -> Any:
            if not tf_imported:
                raise ImportError("TensorFlow is not installed.")
            
            # Convert inputs to NumPy arrays if they are TF tensors
            numpy_inputs = []
            for inp in inputs:
                if hasattr(inp, 'numpy'): # Check if it's a TF tensor
                    numpy_inputs.append(inp.numpy())
                else:
                    numpy_inputs.append(inp) # Assume already NumPy or compatible
            
            # Execute the compiled model (JITExecutor handles NumPy arrays)
            # The JITExecutor.execute expects a list of input arrays.
            raw_outputs = jit_executor.execute(func_name_to_call, numpy_inputs)
            
            # Convert outputs back to TensorFlow tensors
            # The structure of raw_outputs depends on the JITExecutor and model.
            # It might be a single NumPy array or a list/tuple of them.
            if isinstance(raw_outputs, (list, tuple)):
                return [tf.convert_to_tensor(out) for out in raw_outputs]
            else:
                return tf.convert_to_tensor(raw_outputs)

        return compiled_model_executor

# Example Usage (for direct testing if TF is installed)
if __name__ == "__main__" and tf_imported:
    logging.basicConfig(level=logging.INFO)
    
    # Simple Keras model
    def create_simple_model():
        inputs = tf.keras.Input(shape=(10,))
        x = tf.keras.layers.Dense(20, activation='relu', name='dense_1')(inputs)
        outputs = tf.keras.layers.Dense(5, activation='softmax', name='dense_2')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SimpleKerasModel")
        return model

    keras_model = create_simple_model()
    example_input_tensor = tf.random.uniform(shape=(1, 10), dtype=tf.float32)

    # Test with default compiler (host detection)
    try:
        tf_frontend = TensorFlowFrontend()
        logger.info("--- Testing TensorFlowFrontend Import ---")
        mlir_output = tf_frontend.import_model(keras_model, example_inputs=example_input_tensor)
        logger.info(f"MLIR (TF Dialect) from Keras model (first 500 chars):\n{mlir_output[:500]}...")
        assert "tf.Const" in mlir_output or "tf.Identity" in mlir_output # Check for TF ops

        logger.info("--- Testing TensorFlowFrontend Compile to LLVM ---")
        compilation_res = tf_frontend.compile(keras_model, example_inputs=example_input_tensor, output_format="llvm")
        llvm_ir = compilation_res.artifact
        logger.info(f"LLVM IR (first 500 chars):\n{llvm_ir[:500]}...")
        assert "target triple" in llvm_ir # Basic check for LLVM IR

        # Conceptual JIT test
        logger.info("--- Testing TensorFlowFrontend JIT Compilation (Conceptual) ---")
        try:
            jit_function = tf_frontend.jit_compile(keras_model, example_inputs=example_input_tensor)
            # To actually run this, the JITExecutor would need to correctly map function names and signatures
            # from the LLVM IR generated from TensorFlow MLIR.
            logger.info(f"JIT function created: {jit_function}")
            # conceptual_output = jit_function(example_input_tensor)
            # logger.info(f"Conceptual JIT output: {conceptual_output}")
        except Exception as e:
            logger.warning(f"Conceptual JIT test for TensorFlow failed or was skipped: {e}")

    except Exception as e:
        logger.error(f"TensorFlowFrontend example failed: {e}", exc_info=True)
        logger.error("This might be due to TensorFlow, MLIR, or LLVM setup issues.")

else:
    if not tf_imported:
        print("TensorFlow not installed. Skipping TensorFlowFrontend example.") 