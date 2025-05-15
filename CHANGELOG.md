# Changelog

All notable changes to OpenCompiler will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Enhanced Apple Silicon vectorization optimization passes in the MLIR pipeline
- Expanded PyTorch frontend to support more operator conversions
- Initial JAX frontend support for basic JAX models and computations
- Improved documentation across the project
- Resource management system for balancing CPU/GPU/ANE usage
- Basic Metal Performance Shaders (MPS) integration for Apple Silicon

### Fixed
- MLIR dialect registration issues on certain platforms
- Memory leak in JIT executor when handling large models
- CPU feature detection on ARM64 systems

## [0.1.0] - 2023-08-15

### Added
- Initial project structure and architecture
- Core compiler infrastructure with MLIR-based optimization pipeline
- CPU target architecture detection (x86-64, ARM64, Apple Silicon)
- Basic PyTorch frontend for simple models
- JIT execution for compiled models
- Documentation, including README, ARCHITECTURE, and CONTRIBUTING guides
- CI/CD integration for automated testing
- Example models and usage patterns 