#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# README.md is preferred for long description by pyproject.toml with Poetry
# This setup.py is more for compatibility or direct setup.py usage.
readme_path = "README.md"
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "OpenCompiler: AI Compiler Engineering Platform for CPU-Optimized ML Workloads. See repository for details."

setup(
    name="opencompiler", # Changed from aicompiler
    version="0.1.0", # Version should ideally be synced with pyproject.toml
    author="The OpenCompiler Team", # Changed author
    author_email="contact@opencompiler.dev", # Changed email
    description="OpenCompiler: An AI Compiler Engineering Platform for CPU-Optimized ML Workloads",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/opencompiler", # Changed URL
    packages=find_packages(include=["aicompiler", "aicompiler.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Compilers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
    ],
    python_requires=">=3.8,<3.12",
    install_requires=[
        "numpy>=1.21.0",
        "llvmlite>=0.40.0",
        "py-cpuinfo>=9.0.0",
        "psutil>=5.9.0",
        # Note: torch, jax, onnx are optional and better handled via extras
        # or by poetry from pyproject.toml. Listing core ones here.
    ],
    extras_require={
        "torch": ["torch>=2.0.0"],
        "jax": ["jax>=0.4.10", "jaxlib>=0.4.10"],
        "onnx": ["onnx>=1.14.0", "onnxoptimizer>=0.3.10"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.10.0",
        ],
    },
    # If you have C++ extensions built via CMake and want to package them using
    # scikit-build or setuptools-cmake, this file would need more setup.
    # For now, C++ is handled separately by Makefile/CMake for conceptual parts.
) 