# Contributing to OpenCompiler

First off, thank you for considering contributing to OpenCompiler! We aim to build a powerful and flexible AI compiler platform, and we welcome contributions from the community.

## How to Contribute

We are in the early stages of development. Currently, the best way to contribute is by:

1.  **Testing**: Clone the repository, set up the environment (see `README.md`), and try running the examples. Report any issues you encounter.
2.  **Feedback**: Provide feedback on the architecture (`ARCHITECTURE.md`), existing code, or features you'd like to see.
3.  **Documentation**: Improvements to documentation are always welcome.
4.  **Code Contributions (Future)**: As the project matures, we will establish clearer guidelines for code contributions, including:
    *   Issue tracking (currently via GitHub Issues if the repo is public).
    *   Pull Request process.
    *   Coding standards (e.g., based on Black, Flake8, iSort for Python).
    *   Testing requirements for new code.

## Setting up a Development Environment

Refer to the `README.md` for detailed installation and setup instructions. Key steps usually involve:

```bash
# 1. Clone the repository
# git clone https://github.com/llamasearchai/opencompiler.git
# cd opencompiler

# 2. Install system dependencies (LLVM, MLIR, CMake, etc.)
# On macOS, ./install_mac.sh can help.

# 3. Set up Python environment and install dependencies
poetry install --all-extras --with dev

# 4. Activate the environment
poetry shell

# 5. Build C++ components (if working on them)
make build-cpp
```

## Reporting Bugs

-   Please check existing issues to see if your bug has already been reported.
-   Provide as much detail as possible: OS, Python version, OpenCompiler version (if applicable), steps to reproduce, expected behavior, and actual behavior.
-   Include relevant logs or error messages.

## Suggesting Enhancements

-   We are open to suggestions for new features or improvements to existing ones.
-   Please provide a clear and detailed explanation of the enhancement and its potential benefits.

## Code Style

-   **Python**: We aim to follow PEP 8. Code is formatted with Black and iSort. Linting is done with Flake8 and MyPy. You can use `make format` and `make lint`.
-   **C++** (for future contributions): We will establish C++ coding standards (e.g., based on Google C++ Style Guide or LLVM Coding Standards).

## Future: Pull Request Process (Placeholder)

1.  Fork the repository.
2.  Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature-name`).
3.  Make your changes. Ensure you add or update tests as appropriate.
4.  Run `make lint` and `make test` to ensure all checks pass.
5.  Commit your changes with a clear and descriptive commit message.
6.  Push your branch to your fork (`git push origin feature/your-feature-name`).
7.  Open a Pull Request against the main OpenCompiler repository.

We look forward to your contributions! 