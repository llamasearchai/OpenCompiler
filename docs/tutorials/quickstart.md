# OpenCompiler Quick Start Guide

This guide will help you get started with OpenCompiler quickly, showing the basic concepts and usage patterns. After following this guide, you'll have a basic understanding of how to use OpenCompiler to optimize ML models for CPU execution.

## Prerequisites

- Ensure OpenCompiler is properly installed (see [Installation Guide](installation.md))
- Basic familiarity with ML frameworks (PyTorch, JAX, or ONNX)
- Python 3.8+

## Hello World: Basic MLIR Compilation

Let's start with a simple "Hello World" example that demonstrates compiling a basic MLIR module:

```python
from aicompiler import CPUCompiler, TargetArch
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Create a compiler instance - it will auto-detect your CPU architecture
compiler = CPUCompiler()
print(f"Using target architecture: {compiler.target_arch.name}")

# Define a simple MLIR module with an add function
mlir_module = """
module {
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %res = arith.addf %arg0, %arg1 : f32
    return %res : f32
  }
}
"""

# Compile the module to LLVM IR
llvm_ir = compiler.compile(mlir_module, output_format="llvm")
print("\nLLVM IR (first 200 chars):\n", llvm_ir[:200])

# You can also compile to object code or assembly
# object_code = compiler.compile(mlir_module, output_format="object")
# assembly = compiler.compile(mlir_module, output_format="assembly")
```

Save this to a file named `hello_world.py` and run it:

```bash
python hello_world.py
```

You should see output showing:
1. The detected CPU architecture
2. The beginning of the generated LLVM IR

## Optimizing a PyTorch Model

Next, let's optimize a simple PyTorch model using the PyTorch frontend:

```python
import torch
from aicompiler import PyTorchFrontend
import time

# Define a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create model and sample input
model = SimpleModel()
model.eval()  # Set to evaluation mode
input_tensor = torch.randn(1, 128)  # Batch size 1, 128 features

# Create the PyTorch frontend
frontend = PyTorchFrontend()
print(f"PyTorch Frontend using target: {frontend.compiler.target_arch.name}")

# Benchmark native PyTorch execution
def benchmark_pytorch():
    start = time.time()
    for _ in range(1000):
        with torch.no_grad():
            output = model(input_tensor)
    end = time.time()
    return (end - start) / 1000  # Average time per inference

# JIT Compile the model with OpenCompiler
compiled_model = frontend.jit_compile(model, input_tensor)

# Benchmark the compiled model
def benchmark_compiled():
    start = time.time()
    for _ in range(1000):
        output = compiled_model(input_tensor)
    end = time.time()
    return (end - start) / 1000  # Average time per inference

# Run benchmarks
pytorch_time = benchmark_pytorch()
compiled_time = benchmark_compiled()

print(f"\nPyTorch native execution: {pytorch_time*1000:.3f} ms per inference")
print(f"OpenCompiler execution: {compiled_time*1000:.3f} ms per inference")
print(f"Speedup: {pytorch_time/compiled_time:.2f}x")
```

Save this to a file named `pytorch_example.py` and run it:

```bash
python pytorch_example.py
```

## Using OpenCompiler with JAX

Here's how to use OpenCompiler with a JAX model:

```python
import jax
import jax.numpy as jnp
from aicompiler import JAXFrontend
import time

# Define a simple JAX function
def jax_model(x):
    w1 = jnp.ones((128, 64))  # Weight matrix 1
    b1 = jnp.zeros(64)        # Bias 1
    w2 = jnp.ones((64, 10))   # Weight matrix 2
    b2 = jnp.zeros(10)        # Bias 2
    
    # Layer 1
    y = jnp.dot(x, w1) + b1
    y = jnp.maximum(0, y)  # ReLU
    
    # Layer 2
    y = jnp.dot(y, w2) + b2
    return y

# Create sample input
input_data = jnp.ones((1, 128))

# JIT compile with native JAX
jax_compiled = jax.jit(jax_model)

# Create JAX frontend
frontend = JAXFrontend()
print(f"JAX Frontend using target: {frontend.compiler.target_arch.name}")

# Compile with OpenCompiler
opencompiler_compiled = frontend.jit_compile(jax_model, input_data)

# Benchmark JAX native
def benchmark_jax():
    start = time.time()
    for _ in range(1000):
        output = jax_compiled(input_data)
    end = time.time()
    return (end - start) / 1000

# Benchmark OpenCompiler
def benchmark_opencompiler():
    start = time.time()
    for _ in range(1000):
        output = opencompiler_compiled(input_data)
    end = time.time()
    return (end - start) / 1000

# Run benchmarks
jax_time = benchmark_jax()
opencompiler_time = benchmark_opencompiler()

print(f"\nJAX native execution: {jax_time*1000:.3f} ms per inference")
print(f"OpenCompiler execution: {opencompiler_time*1000:.3f} ms per inference")
print(f"Speedup: {jax_time/opencompiler_time:.2f}x")
```

## Working with ONNX Models

You can also optimize ONNX models:

```python
import onnx
import numpy as np
from aicompiler import ONNXFrontend
import time

# Load an ONNX model (you need to have an ONNX model file)
# For this example, we'll create a simple one
from onnx import helper, TensorProto
import onnx.numpy_helper

# Create an ONNX model
def create_simple_onnx_model():
    # Create nodes (operations)
    node1 = helper.make_node(
        'MatMul',
        inputs=['input', 'weight1'],
        outputs=['hidden']
    )
    
    node2 = helper.make_node(
        'Relu',
        inputs=['hidden'],
        outputs=['hidden_relu']
    )
    
    node3 = helper.make_node(
        'MatMul',
        inputs=['hidden_relu', 'weight2'],
        outputs=['output']
    )
    
    # Create inputs
    input = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 128])
    weight1 = helper.make_tensor_value_info('weight1', TensorProto.FLOAT, [128, 64])
    weight2 = helper.make_tensor_value_info('weight2', TensorProto.FLOAT, [64, 10])
    
    # Create outputs
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 10])
    
    # Create initializers
    weight1_data = np.ones((128, 64), dtype=np.float32)
    weight2_data = np.ones((64, 10), dtype=np.float32)
    
    weight1_init = onnx.numpy_helper.from_array(weight1_data, name='weight1')
    weight2_init = onnx.numpy_helper.from_array(weight2_data, name='weight2')
    
    # Create graph
    graph = helper.make_graph(
        nodes=[node1, node2, node3],
        name='simple_model',
        inputs=[input],
        outputs=[output],
        initializer=[weight1_init, weight2_init]
    )
    
    # Create model
    model = helper.make_model(
        graph,
        producer_name='onnx-example'
    )
    
    return model

# Create model and save it
model = create_simple_onnx_model()
onnx.save(model, 'simple_model.onnx')

# Create ONNX frontend
frontend = ONNXFrontend()
print(f"ONNX Frontend using target: {frontend.compiler.target_arch.name}")

# Prepare input data
input_data = np.ones((1, 128), dtype=np.float32)

# Compile the model
compiled_model = frontend.jit_compile('simple_model.onnx', input_data)

# Run the compiled model
output = compiled_model(input_data)
print(f"Output shape: {output.shape}")
```

## Advanced Usage: Custom Optimization Pipeline

You can customize the optimization pipeline for specific architectures:

```python
from aicompiler import CPUCompiler, TargetArch

# Create a compiler with specific architecture and optimization level
compiler = CPUCompiler(target_arch=TargetArch.APPLE_SILICON, optimization_level=3)

# Define a custom optimization pipeline
custom_pipeline = [
    "canonicalize",
    "cse",
    "linalg-fusion",
    "arm-vectorize",
    "apple-ane-optimize",  # Apple-specific optimizations
    "lower-affine",
    "convert-scf-to-cf"
]

# Compile with custom pipeline
def compile_with_custom_pipeline(mlir_module):
    # Use internal method to set custom pipeline
    pipeline_str = ",".join(custom_pipeline)
    
    # Get the underlying MLIR context and module
    with compiler.context:
        # Parse the input IR
        module = Module.parse(mlir_module, compiler.context)
        
        # Create custom pass manager
        from mlir.passmanager import PassManager
        pm = PassManager.parse(pipeline_str, compiler.context)
        
        # Run the custom pipeline
        pm.run(module)
        
        # Lower to LLVM IR
        return compiler._lower_to_llvm(module)
```

## Working with the Resource Manager

For optimal performance on multi-core systems:

```python
from aicompiler.runtime.resource_manager import ResourceManager, ResourceType

# Create a resource manager
resource_mgr = ResourceManager()

# Allocate resources for a task
task_id = "matrix_multiply_1"
resources = {
    ResourceType.CPU: 4  # Allocate 4 CPU cores
}

# Try to allocate resources
if resource_mgr.allocate(task_id, resources):
    try:
        # Run your computation
        print("Running computation with 4 CPU cores")
        # ...your code here...
    finally:
        # Release resources when done
        resource_mgr.release(task_id)
else:
    print("Not enough resources available")
```

## Next Steps

Now that you've seen the basics, you might want to explore:

- [Apple Silicon Optimizations](../apple_silicon/apple_silicon_optimizations.md) - Learn about specialized optimizations for Apple M-series chips
- [Performance Tuning Guide](performance_tuning.md) - Tips for getting the best performance
- [API Reference](../api/reference.md) - Detailed API documentation
- [Architecture Overview](../architecture/overview.md) - Understand the internals of OpenCompiler 