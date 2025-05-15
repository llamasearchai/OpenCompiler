# OpenCompiler Installation Guide

This guide provides detailed installation instructions for OpenCompiler on different platforms. OpenCompiler has specific system requirements due to its dependencies on LLVM/MLIR and other libraries.

## System Requirements

- **Python**: 3.8 - 3.11
- **LLVM/MLIR**: Version 15.0 or newer
- **C++ Compiler**: 
  - macOS: Clang 13+ (included with Xcode)
  - Linux: GCC 10+ or Clang 13+
- **Build Tools**: CMake 3.14+, Ninja (recommended)
- **Poetry**: For Python package management

## Quick Install

### macOS (Apple Silicon or Intel)

We provide a convenient script for macOS users that handles most dependencies and setup:

```bash
# Clone the repository
git clone https://github.com/yourusername/opencompiler.git
cd opencompiler

# Make the install script executable
chmod +x install_mac.sh

# Run the installer
./install_mac.sh
```

The script will:
1. Install Homebrew if not present
2. Install required dependencies (LLVM, CMake, Ninja, Python)
3. Set up environment variables
4. Install Python dependencies with Poetry

### Linux Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/opencompiler.git
cd opencompiler

# Install system dependencies (Ubuntu/Debian example)
sudo apt update
sudo apt install -y cmake ninja-build python3-pip python3-venv

# Install LLVM/MLIR development packages
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 15

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies with Poetry
poetry install --all-extras --with dev

# Build C++ components (if any)
make build-cpp
```

## Detailed Installation Guide

### 1. Installing LLVM/MLIR

#### macOS with Homebrew

```bash
brew install llvm@15
```

After installation, add LLVM to your PATH and set required environment variables:

```bash
echo 'export PATH="/opt/homebrew/opt/llvm@15/bin:$PATH"' >> ~/.zshrc
echo 'export LDFLAGS="-L/opt/homebrew/opt/llvm@15/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/opt/homebrew/opt/llvm@15/include"' >> ~/.zshrc
source ~/.zshrc
```

#### Linux (Ubuntu/Debian)

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 15

# Install development packages
sudo apt-get install -y libllvm-15-ocaml-dev libllvm15 llvm-15 llvm-15-dev llvm-15-doc llvm-15-examples llvm-15-runtime
sudo apt-get install -y libmlir-15-dev mlir-15-tools
```

### 2. Installing Python Dependencies

We use Poetry for managing Python dependencies. First, install Poetry:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the project dependencies:

```bash
cd opencompiler
poetry install --all-extras --with dev
```

To activate the Poetry environment:

```bash
poetry shell
```

### 3. Building C++ Components

OpenCompiler includes some C++ components that need to be built:

```bash
# Create and navigate to build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build
cmake --build . --parallel $(nproc)
```

Alternatively, you can use the Makefile:

```bash
make build-cpp
```

### 4. Verifying Installation

To verify that OpenCompiler is correctly installed, run one of the example scripts:

```bash
# Activate Poetry environment if not already activated
poetry shell

# Run a simple example
python examples/simple_example.py
```

## Framework-Specific Setup

### PyTorch Integration

```bash
# Install PyTorch with Poetry
poetry add torch
```

For Apple Silicon users, you might want to install PyTorch with MPS support:

```bash
# Install PyTorch with MPS support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

### JAX Integration

```bash
# Install JAX with Poetry
poetry add jax jaxlib
```

For Apple Silicon users, install the CPU version of JAX:

```bash
pip install --upgrade "jax[cpu]"
```

### ONNX Integration

```bash
# Install ONNX with Poetry
poetry add onnx onnxoptimizer
```

## Troubleshooting

### Common Issues

#### LLVM/MLIR Not Found

If you encounter errors related to LLVM or MLIR not being found:

```bash
# Set environment variables explicitly
export LLVM_CONFIG=/path/to/llvm-config
export MLIR_DIR=/path/to/mlir/lib/cmake/mlir
```

#### Poetry Installation Issues

If you encounter issues with Poetry:

```bash
# Update Poetry
poetry self update

# Clear Poetry cache
poetry cache clear --all .
```

#### Build Failures

If C++ component builds fail:

```bash
# Remove build directory and start fresh
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make VERBOSE=1
```

### Platform-Specific Issues

#### Apple Silicon-Specific Issues

On Apple M-series chips, you might need to:

```bash
# Install Rosetta for some Intel-only packages
softwareupdate --install-rosetta

# Force native arm64 architecture
export CMAKE_OSX_ARCHITECTURES=arm64
```

#### Linux Distribution Variations

Different Linux distributions might require slightly different packages:

- **Fedora/CentOS/RHEL**:
  ```bash
  sudo dnf install llvm-devel llvm-static mlir-devel
  ```

- **Arch Linux**:
  ```bash
  sudo pacman -S llvm clang mlir cmake ninja
  ```

## Next Steps

After installation, check out the [Quick Start Guide](quickstart.md) to begin using OpenCompiler. 