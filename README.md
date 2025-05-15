# OpenCompiler: ML Compiler Engineering Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/opencompiler/ci.yml?branch=main)](https://github.com/yourusername/opencompiler/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://yourusername.github.io/opencompiler)
[![Downloads](https://img.shields.io/github/downloads/yourusername/opencompiler/total.svg)](https://github.com/yourusername/opencompiler/releases)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<div align="center">
  <img src="docs/assets/images/logo.png" alt="OpenCompiler Logo" width="300">
  <h3>High-Performance ML Inference on CPUs</h3>
  <h4>Optimized for Apple Silicon & Modern CPU Architectures</h4>
</div>

OpenCompiler is a cutting-edge compiler engineering platform designed to optimize machine learning workloads on modern CPU architectures. The platform leverages MLIR/LLVM infrastructure to transform ML models from PyTorch, JAX, and ONNX into highly efficient CPU-specific machine code, with particular focus on Apple Silicon (M-series) and multi-architecture CPU support.

## 🚀 Key Features

- **Advanced Compiler Infrastructure**: LLVM/MLIR-based pipeline with sophisticated optimization passes
- **Multi-Architecture Support**:
  - **Apple Silicon**: Specialized optimizations for M1/M2/M3-series chips (2-4× speedups)
  - **x86-64**: AVX/AVX2/AVX-512 vectorization
  - **ARM64**: NEON/SVE vectorization
- **Framework Integration**: Seamless frontends for PyTorch, JAX, and ONNX
- **JIT & AOT Compilation**: Dynamic JIT and ahead-of-time compilation options
- **Heterogeneous Execution**: Conceptual support for offloading to specialized hardware (MPS, ANE)
- **Comprehensive Documentation**: Architecture guides, API references, and optimization tutorials

## 📈 Performance Highlights

| Model | Platform | OpenCompiler Speedup | Memory Reduction |
|-------|----------|----------------------|------------------|
| ResNet-50 | Apple M2 Ultra | 2.3× | 7% |
| BERT-Base | Apple M1 Pro | 2.5× | 6.5% |
| ResNet-50 | Intel i9-12900K | 1.8× | 6.4% |
| BERT-Base | AMD Ryzen 7950X | 1.9× | 5.7% |

[See detailed benchmarks](BENCHMARKS.md)

## 🛠️ Apple Silicon Optimizations

OpenCompiler implements several Apple Silicon-specific optimizations:

1. **Cache-Aware Tiling**: Custom-tuned for M-series cache hierarchy
2. **NEON Vectorization**: Fully utilizing 128-bit NEON units with optimized instruction selection
3. **Performance/Efficiency Core Distribution**: Strategic workload distribution across P/E cores
4. **Metal & ANE Integration Pathways**: Conceptual support for GPU & Neural Engine acceleration

[Detailed Apple Silicon optimization documentation](docs/apple_silicon/apple_silicon_optimizations.md)

## 📦 Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/opencompiler.git
cd opencompiler

# macOS quick install
chmod +x install_mac.sh
./install_mac.sh

# Install Python dependencies with Poetry
poetry install --all-extras --with dev
```

[Detailed installation guide](docs/tutorials/installation.md)

## 🚀 Getting Started

```python
from aicompiler import CPUCompiler, TargetArch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create compiler (auto-detects architecture)
compiler = CPUCompiler()
print(f"Using target: {compiler.target_arch.name}")

# Simple MLIR module
mlir_module = """
module {
  func.func @add(%arg0: f32, %arg1: f32) -> f32 {
    %res = arith.addf %arg0, %arg1 : f32
    return %res : f32
  }
}
"""

# Compile to LLVM IR
llvm_ir = compiler.compile(mlir_module, output_format="llvm")
print(f"Generated LLVM IR (first 100 chars): {llvm_ir[:100]}")
```

### PyTorch Integration

```python
import torch
from aicompiler import PyTorchFrontend

# Define PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10)
)
model.eval()

# Sample input
input_tensor = torch.randn(1, 128)

# Compile with OpenCompiler
frontend = PyTorchFrontend()
compiled_model = frontend.jit_compile(model, input_tensor)

# Run compiled model
output = compiled_model(input_tensor)
```

[More examples](docs/tutorials/quickstart.md)

## 📋 Project Structure

```
aicompiler/
├── compiler/         # Core compiler infrastructure
├── integrations/     # ML framework frontends
├── runtime/          # Execution engines (JIT & AOT)
├── src/              # C++ implementation of critical components
└── tests/            # Comprehensive test suite
```

[Full architecture overview](ARCHITECTURE.md)

## 📊 Benchmarks

OpenCompiler achieves significant speedups across various ML workloads:

- **Matrix Operations**: Up to 4.5× faster with cache-aware tiling
- **Convolutional Layers**: 2.2-2.8× speedup with SIMD vectorization
- **Transformer Blocks**: 1.8-2.6× faster with fused operations
- **Memory Usage**: 5-10% reduction in memory footprint

[Detailed benchmark results](BENCHMARKS.md)

## 🗺️ Roadmap

See our detailed roadmap for upcoming features and improvements:

- **v0.2**: Enhanced Apple Silicon optimizations, expanded PyTorch operator coverage
- **v0.3**: Dynamic shape support, auto-tuning, advanced fusion patterns
- **v0.4**: Heterogeneous execution, improved MPS/ANE integration pathways
- **v1.0**: Production-ready release with comprehensive documentation and examples

[Full roadmap](ROADMAP.md)

## 🤝 Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details on how to get involved.

## 📚 Documentation

- [API Reference](docs/api/reference.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Optimization Techniques](docs/architecture/optimizations.md)
- [Performance Tuning](docs/tutorials/performance_tuning.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔍 Citation

If you use OpenCompiler in your research, please cite:

```bibtex
@software{opencompiler2023,
  author = {OpenCompiler Team},
  title = {OpenCompiler: An ML Compiler Engineering Platform},
  url = {https://github.com/yourusername/opencompiler},
  version = {0.1.0},
  year = {2023},
}
```

## Resource Management

```
import threading
import psutil
import os
from typing import Dict, Optional
from enum import Enum, auto

class ResourceType(Enum):
    CPU = auto()
    GPU = auto()
    ANE = auto()  # Apple Neural Engine

class ResourceManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._resources: Dict[ResourceType, int] = {
            ResourceType.CPU: os.cpu_count() or 1,
            ResourceType.GPU: self._detect_gpu_count(),
            ResourceType.ANE: self._detect_ane_availability()
        }
        self._allocations: Dict[str, Dict[ResourceType, int]] = {}

    def _detect_gpu_count(self) -> int:
        try:
            import torch
            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _detect_ane_availability(self) -> int:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                import torch
                return 1 if torch.backends.mps.is_available() else 0
            except ImportError:
                pass
        return 0

    def allocate(self, task_id: str, resources: Dict[ResourceType, int]) -> bool:
        with self._lock:
            for r_type, count in resources.items():
                if self._resources.get(r_type, 0) < count:
                    return False
            for r_type, count in resources.items():
                self._resources[r_type] -= count
            self._allocations[task_id] = resources
            return True

    def release(self, task_id: str) -> None:
        with self._lock:
            if task_id in self._allocations:
                for r_type, count in self._allocations[task_id].items():
                    self._resources[r_type] += count
                del self._allocations[task_id] 