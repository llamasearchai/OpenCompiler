# OpenCompiler Platform Architecture

## 1. Overview

The OpenCompiler Platform is a modular, extensible system designed to compile and optimize machine learning (ML) workloads for various CPU architectures, with a special emphasis on Apple Silicon (M-series chips) and other ARM64 processors, alongside traditional x86-64 CPUs.

Its primary goal is to take high-level ML model representations from popular frameworks (PyTorch, JAX, ONNX) and transform them into highly efficient, CPU-specific machine code. This involves leveraging MLIR (Multi-Level Intermediate Representation) as its core compiler infrastructure.

## 2. Core Components & Data Flow

The platform can be visualized as a pipeline:

```mermaid
graph TD
    A[ML Framework Model (PyTorch, JAX, ONNX)] --> B{Framework Frontend};
    B --> C[MLIR: High-Level Dialect (e.g., torch, mhlo, onnx)];
    C --> D{MLIR: Generic Optimizations (Canonicalization, CSE, Fusion)};
    D --> E{MLIR: CPU-Agnostic ML Optimizations (e.g., Linalg on Tensors)};
    E --> F{MLIR: Architecture-Specific Lowering & Optimizations};
    F --> G[MLIR: Vector/SCF/Affine Dialects];
    G --> H{MLIR: LLVM Dialect Conversion};
    H --> I[LLVM IR];
    I --> J{LLVM: Code Generation & Optimizations (Target Specific)};
    J --> K[Optimized CPU Machine Code (.o, .so, JIT)];

    subgraph "CPU Targets"
        K --> L[x86-64 (AVX2/512)];
        K --> M[ARM64 (NEON/SVE)];
        K --> N[Apple Silicon (NEON, ANE/Metal Placeholders)];
    end

    subgraph "Execution"
        K --> O{Runtime Environment};
        O --> P[JIT Execution (JITExecutor)];
        O --> Q[AOT Execution (CPURuntimeExecutor)];
    end
```

### 2.1. Framework Integration Layer

-   **Purpose**: To ingest models from popular ML frameworks and convert them into a common MLIR dialect suitable for the OpenCompiler system.
-   **Components**:
    -   `PyTorchFrontend`: Converts PyTorch models (via `torch.fx` or `torch.jit`) into the MLIR `torch` dialect (or directly to higher-level Linalg).
    -   `JAXFrontend`: Converts JAX computations (via JAX Pytrees and JAXPR/StableHLO) into the MLIR `mhlo` (or `stablehlo`) dialect.
    -   `ONNXFrontend`: Parses ONNX models and converts them into the MLIR `onnx` dialect.
-   **Output**: MLIR modules in a framework-specific or high-level ML dialect.

### 2.2. Core Compiler Infrastructure (MLIR-based)

-   **Purpose**: To perform a series of transformations and optimizations on the MLIR representation.
-   **Key MLIR Dialects Used**:
    -   **Framework-specific dialects**: `torch`, `mhlo`, `onnx` (as input).
    -   **High-level ML/Numeric dialects**: `linalg` (for tensor algebra), `tosa`.
    -   **Looping and Control Flow**: `scf` (Structured Control Flow), `cf` (Control Flow).
    -   **Memory and Buffers**: `memref`.
    -   **Vectorization**: `vector`.
    -   **Affine**: `affine` (for polyhedral loop optimizations, if applicable).
    -   **LLVM Dialect**: The final MLIR stage before LLVM IR.
-   **Optimization Passes (Conceptual examples)**:
    -   **Generic IR Cleanup**: Canonicalization, Common Subexpression Elimination (CSE), Dead Code Elimination (DCE), Inlining.
    -   **High-Level Optimizations**: Operator fusion (e.g., fusing Conv-BN-ReLU), bufferization, tiling for cache locality using `linalg`.
    -   **CPU-Agnostic ML Optimizations**: Transformations on `linalg` named ops or generic ops.

### 2.3. Optimization Engine (Architecture-Specific)

-   **Purpose**: To apply optimizations tailored to the specific microarchitecture of the target CPU.
-   **Components & Strategies**:
    -   **`TargetArch` Enum**: Identifies the target CPU, enabling dispatch to specific optimization pipelines.
    -   **x86-64 Optimizations**:
        -   Lowering to `vector` dialect ops that map well to AVX/AVX2/AVX-512 instructions.
        -   Passes for instruction selection and scheduling beneficial for x86.
    -   **ARM64 (Generic) Optimizations**:
        -   NEON Vectorization: Transforming operations to use 128-bit NEON SIMD units via the `vector` dialect or ARM NEON MLIR dialect (if available).
        -   SVE Vectorization (if target supports SVE): Similar to NEON but for scalable vectors.
    -   **Apple Silicon (M-series) Specific Optimizations**:
        -   **NEON**: Full utilization of ARMv8 NEON capabilities.
        -   **ANE (Apple Neural Engine) Integration (Conceptual)**: 
            -   Passes to identify subgraphs compatible with ANE.
            -   Lowering these subgraphs to a hypothetical ANE dialect or runtime calls that invoke ANE routines (e.g., via CoreML or private frameworks if ever exposed).
            -   The `apple-ane-optimize` pass in `CPUCompiler` is a placeholder for this.
        -   **Metal Performance Shaders (MPS) Integration (Conceptual for direct MLIR->Metal)**:
            -   Identifying compute-intensive ops (e.g., large matmuls, convolutions) that could benefit from GPU execution via MPS.
            -   Lowering to MLIR `gpu` dialect ops, which could then (conceptually) be translated to Metal Shading Language and executed via MPS routines.
            -   The PyTorch frontend checks for MPS compatibility for PyTorch models, hinting at potential offload paths.
        -   **Unified Memory**: Optimizations aware of the unified memory architecture to minimize data copies between CPU and GPU (if MPS is used).
    -   **C++ Placeholders**: `arm_vectorizer.cpp` is a placeholder for custom C++ MLIR passes that could implement some of these low-level transformations.

### 2.4. LLVM IR Conversion & Code Generation

-   **Purpose**: Convert the optimized MLIR (in LLVM dialect) to LLVM IR, and then to machine code.
-   **Process**:
    -   MLIR modules in the `llvm` dialect are passed to `llvmlite` (Python bindings for LLVM) or a direct LLVM C++ API.
    -   LLVM performs its own rich set of optimization passes (instruction combining, loop optimizations, register allocation, etc.) tailored to the specific CPU target (e.g., `apple-m1`, `haswell`, `znver2`).
    -   LLVM's backend generates the final machine code (assembly, object file, or JIT-ready code).

### 2.5. Runtime Environment

-   **Purpose**: To execute the compiled code, either JIT-compiled or loaded from AOT artifacts.
-   **Components**:
    -   **`JITExecutor`**: 
        -   Takes LLVM IR (as a string).
        -   Uses `llvmlite` to create an MCJIT execution engine.
        -   Compiles the LLVM IR in memory to executable machine code.
        -   Provides an `execute` method to run named functions from the JITted module, handling data marshalling (NumPy to/from C-compatible types via `ctypes`).
    -   **`CPURuntimeExecutor` (AOT)**:
        -   Loads pre-compiled shared libraries (`.so`, `.dylib`).
        -   Uses `ctypes` to find and call exported C-ABI functions within these libraries.
        -   Manages different execution paths (e.g., standard, `_metal`, `_ane` suffixed functions) conceptually for Apple Silicon.
        -   Includes benchmarking capabilities.

## 3. Data Flow Summary

1.  **Model Ingestion**: User provides a model (PyTorch, JAX, ONNX) and example inputs.
2.  **Frontend Conversion**: The appropriate framework frontend (e.g., `PyTorchFrontend`) traces/converts the model into an MLIR module (e.g., `torch` dialect, then to `linalg`/`mhlo`).
3.  **MLIR Optimizations (Core Compiler)**: 
    -   The `CPUCompiler` takes the MLIR module string.
    -   It applies a sequence of MLIR passes based on the `TargetArch` and optimization level.
    -   This includes generic passes, ML-specific passes, and architecture-specific passes (e.g., `arm-vectorize`, `apple-ane-optimize` conceptual pass).
4.  **Lowering to LLVM**: The optimized MLIR is lowered to the LLVM dialect within MLIR.
5.  **Code Generation (LLVM via `CPUCompiler` / `llvmlite`)**:
    -   For JIT: The LLVM dialect MLIR is converted to an LLVM IR string.
    -   For AOT: The LLVM dialect MLIR is converted to LLVM IR, then to an object file or assembly string.
6.  **Execution**:
    -   **JIT**: `JITExecutor` takes the LLVM IR string, compiles it, and makes functions callable.
    -   **AOT**: `CPURuntimeExecutor` loads a pre-compiled shared library (produced by linking the object file from step 5) and calls its functions.

## 4. Apple Silicon Specifics - Deeper Dive

-   **Detection**: `TargetArch.detect_host()` uses `platform.machine()` and `platform.processor()` to identify Apple Silicon.
-   **Compiler Optimizations**: The `CPUCompiler._get_pipeline_string()` method includes Apple-specific (conceptual) pass names like `apple-ane-optimize` and generic ARM passes like `arm-vectorize` when `target_arch` is `APPLE_SILICON`.
-   **LLVM Targeting**: When compiling to object code or assembly, `llvmlite` is configured with CPU type like `apple-m1` (or similar) and relevant features (`+neon`, etc.) to enable LLVM to generate optimal ARM64 code for M-series chips.
-   **Runtime Dispatch (Conceptual for AOT)**: `CPURuntimeExecutor` has logic to look for functions with `_metal` or `_ane` suffixes, providing a hook for specialized execution paths if such functions were compiled into the AOT artifact.
-   **PyTorch MPS**: The `PyTorchFrontend` checks for MPS availability and model compatibility on Apple Silicon, which can influence its MLIR conversion strategy (e.g., by trying to preserve structures amenable to GPU offload or using MPS-specific conversion passes if they existed in MLIR).

This architecture aims to provide a flexible and powerful platform for ML model compilation, with clear extension points for new frontends, optimization passes, and hardware backends within the OpenCompiler ecosystem. 