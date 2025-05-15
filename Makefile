# Makefile for AICompilerPlatform

.PHONY: all setup install clean test format lint build-cpp docs help

# Default shell
SHELL := /bin/bash

# Python interpreter (uses the one in the environment, hopefully with poetry)
PYTHON := python3
POETRY := poetry

# Build directory for CMake
BUILD_DIR := build

# --- Primary Targets ---
all: setup build-cpp install test

# Setup development environment
setup:
	@echo "+ Setting up Python dependencies with Poetry..."
	@$(POETRY) install --all-extras --with dev
	@echo "+ Poetry environment is set up. Activate with 'poetry shell' or use 'poetry run ...'"

# Install the package
install:
	@echo "+ Installing the aicompiler package..."
	@$(POETRY) install --all-extras # Ensure all optional dependencies are installed
	@echo "+ To build and install C++ components (if any), run 'make build-cpp' and then ensure your Python environment can find them."

# Build C++ components (if any)
# This is a basic CMake build. For more complex projects, you might have Release/Debug types.
build-cpp:
	@echo "+ Building C++ components (if any)..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build . --parallel $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
	@echo "+ C++ build complete. Executables/libraries are in $(BUILD_DIR)."

# Run tests
test:
	@echo "+ Running Python tests..."
	@$(POETRY) run pytest -vv tests/
	@echo "+ To run C++ tests (if defined in CMake), navigate to $(BUILD_DIR) and run 'ctest'"

# Clean build artifacts
clean:
	@echo "+ Cleaning build artifacts and caches..."
	@rm -rf $(BUILD_DIR)
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf *.egg-info
	@rm -rf dist
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -delete
	@echo "+ Clean complete."

# Format code
format:
	@echo "+ Formatting Python code with Black and iSort..."
	@$(POETRY) run black .
	@$(POETRY) run isort .

# Lint code
lint:
	@echo "+ Linting Python code with Flake8 and MyPy..."
	@$(POETRY) run flake8 aicompiler tests examples
	@$(POETRY) run mypy aicompiler

# Generate documentation (placeholder)
docs:
	@echo "+ Generating documentation (placeholder)..."
	@echo "  To implement, consider using Sphinx or MkDocs."

# Display help
help:
	@echo "Available targets:"
	@echo "  all         : Setup, build C++ components, install, and test (Default)"
	@echo "  setup       : Install Python dependencies using Poetry."
	@echo "  install     : Install the Python package using Poetry."
	@echo "  build-cpp   : Build C++ components using CMake."
	@echo "  test        : Run Python tests."
	@echo "  clean       : Remove build artifacts and caches."
	@echo "  format      : Format Python code with Black and iSort."
	@echo "  lint        : Lint Python code with Flake8 and MyPy."
	@echo "  docs        : Generate documentation (placeholder)."
	@echo "  help        : Show this help message." 