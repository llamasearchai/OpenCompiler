#!/bin/bash
# Installation script for OpenCompiler Platform on Mac M3 Max

# Exit on error
set -e

# Print with color
print_info() {
    echo -e "\033[1;34m[INFO_OpenCompiler]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS_OpenCompiler]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR_OpenCompiler]\033[0m $1"
}

# Check if running on macOS
if [ "$(uname)" != "Darwin" ]; then
    print_error "This script is designed for macOS. Please use the appropriate installation method for your OS."
    exit 1
fi

# Check if running on ARM64 (Apple Silicon)
if [ "$(uname -m)" != "arm64" ]; then
    print_error "This script is optimized for Apple Silicon Macs (for OpenCompiler). It may not work correctly on Intel Macs."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    print_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [ "$(uname -m)" = "arm64" ]; then
        print_info "Adding Homebrew to PATH for Apple Silicon in ~/.zprofile"
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# Install dependencies with Homebrew
print_info "Installing dependencies for OpenCompiler (cmake, ninja, llvm@15, python@3.10/3.11)..."
brew update
# Try python@3.11 first, fallback to 3.10 if specific projects require it
brew install cmake ninja llvm@15 python@3.11 || brew install python@3.10

# Set up environment variables for LLVM (ensure this is idempotent)
LLVM_PATH_ARM="/opt/homebrew/opt/llvm@15/bin"
LLVM_PATH_INTEL="/usr/local/opt/llvm@15/bin"
ZSHRC_FILE=~/.zshrc
PROFILE_FILE=~/.zprofile # For login shells on macOS

print_info "Setting up LLVM environment variables in $ZSHRC_FILE and $PROFILE_FILE..."

SETUP_LLVM_ENV() {
    local llvm_bin_path=$1
    local target_file=$2
    grep -q "$llvm_bin_path" "$target_file" || {
        print_info "Adding LLVM from $llvm_bin_path to PATH in $target_file"
        echo "# Added by OpenCompiler install script" >> "$target_file"
        echo "export PATH=\"$llvm_bin_path:\$PATH\"" >> "$target_file"
        echo "export LDFLAGS=\"-L${llvm_bin_path%/bin}/lib\"" >> "$target_file"
        echo "export CPPFLAGS=\"-I${llvm_bin_path%/bin}/include\"" >> "$target_file"
    }
}

if [ "$(uname -m)" = "arm64" ]; then
    SETUP_LLVM_ENV $LLVM_PATH_ARM $ZSHRC_FILE
    SETUP_LLVM_ENV $LLVM_PATH_ARM $PROFILE_FILE
else
    SETUP_LLVM_ENV $LLVM_PATH_INTEL $ZSHRC_FILE
    SETUP_LLVM_ENV $LLVM_PATH_INTEL $PROFILE_FILE
fi

print_info "Please source your shell configuration (e.g., 'source ~/.zshrc') or open a new terminal."

# Install/Update Poetry
if ! command -v poetry &> /dev/null; then
    print_info "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    print_info "Updating Poetry..."
    poetry self update
fi

# Ensure correct Python is used by poetry if system has multiple pythons
# Best to run this in a directory with pyproject.toml that specifies python version
# Or ensure the python3 from brew is first in PATH before running poetry install.

# Install Python dependencies via Poetry
print_info "Installing OpenCompiler Python dependencies using Poetry..."
# This assumes the script is run from the root of the cloned OpenCompiler repository
if [ ! -f pyproject.toml ]; then
    print_error "pyproject.toml not found. Please run this script from the root of the OpenCompiler repository."
    exit 1
fi
poetry install --all-extras --with dev

# Install the package in development mode using Poetry
print_info "Ensuring OpenCompiler (aicompiler package) is installed in editable mode..."
poetry install --all-extras # This should handle it if already specified

print_info "Building C++ components for OpenCompiler (if any)..."
make build-cpp

# Run the simple example to verify installation
print_info "Running the OpenCompiler example to verify installation..."
poetry run python examples/simple_example.py

print_success "OpenCompiler installation and setup completed successfully!"
print_info "Activate the virtual environment with 'poetry shell' to use OpenCompiler."
print_info "To test the PyTorch integration, run: poetry run python examples/pytorch_example.py" 