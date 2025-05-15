"""
AI Compiler Engineering Platform for CPU-Optimized ML Workloads.

A comprehensive compiler engineering platform designed for optimizing ML workloads 
on CPU architectures with special support for Apple Silicon.
"""

__version__ = "0.1.0"

# Core compiler components
from .compiler.core import CPUCompiler, TargetArch, CompilationResult
from .compiler.codegen import CodeGenerator, ISAType, CodeGenerationOptions

# Runtime components
from .runtime.executor import CPURuntimeExecutor
from .runtime.jit_executor import JITExecutor
from .runtime.resource_manager import ResourceManager, ResourceType
from .runtime.scheduler import Task, TaskScheduler

# Framework frontends
from .integrations.pytorch_frontend import PyTorchFrontend
from .integrations.jax_frontend import JAXFrontend
from .integrations.onnx_frontend import ONNXFrontend
from .integrations.tensorflow_frontend import TensorFlowFrontend

# Optimization components
from .optimizations.graph_optimizer import (
    GraphOptimizer,
    PyTorchGraphOptimizer,
    TensorFlowOptimizer,
    JAXGraphOptimizer,
    ONNXGraphOptimizer
)

__all__ = [
    # Compiler
    "CPUCompiler",
    "TargetArch",
    "CompilationResult",
    "CodeGenerator",
    "ISAType",
    "CodeGenerationOptions",
    # Runtime
    "CPURuntimeExecutor",
    "JITExecutor",
    "ResourceManager",
    "ResourceType",
    "Task",
    "TaskScheduler",
    # Integrations
    "PyTorchFrontend",
    "JAXFrontend",
    "ONNXFrontend",
    "TensorFlowFrontend",
    # Optimizations
    "GraphOptimizer",
    "PyTorchGraphOptimizer",
    "TensorFlowOptimizer",
    "JAXGraphOptimizer",
    "ONNXGraphOptimizer",
    # Version
    "__version__"
]

# Basic logging configuration for the library
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler()) 