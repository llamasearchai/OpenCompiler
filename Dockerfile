# Stage 1: Builder environment
# Use a base image with build tools for C++ and Python
# Parameterize for different architectures using BUILDPLATFORM
FROM --platform=${BUILDPLATFORM:-ubuntu:22.04} ubuntu:22.04 AS builder

ARG TARGETPLATFORM
ARG TARGETARCH
ARG BUILDPLATFORM
ARG BUILDARCH

ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies and C++ build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    ca-certificates \
    # Python build dependencies
    python3-dev \
    python3-pip \
    python3-venv \
    # LLVM/MLIR (Example: LLVM 15, adjust version as needed)
    # This setup can be complex and platform-dependent. 
    # For a robust solution, consider pre-built LLVM/MLIR images or meticulous source builds.
    # Here, we use Ubuntu's apt.llvm.org repository as an example.
    wget && \
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 15 all && \
    rm llvm.sh && \
    # Link a generic mlir-opt for easier use if needed (actual path might be /usr/bin/mlir-opt-15)
    # ln -s /usr/bin/mlir-opt-15 /usr/bin/mlir-opt || true && \
    # ln -s /usr/bin/mlir-translate-15 /usr/bin/mlir-translate || true && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
# Pin a version for consistency
ENV POETRY_VERSION=1.7.1 
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VENV="/opt/poetry-venv"
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN python3 -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install --upgrade pip && \
    $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION} && \
    ln -s $POETRY_VENV/bin/poetry $POETRY_HOME/bin/poetry

# Set up the application directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
# poetry.lock is essential for reproducible builds, assume it exists.
COPY poetry.lock /app/ 
COPY CMakeLists.txt /app/
COPY aicompiler /app/aicompiler/
COPY src /app/src/

# Install Python dependencies using Poetry
RUN poetry install --no-interaction --no-ansi --all-extras --with dev

# Build C++ components (conceptual, if any)
RUN mkdir -p build_cpp && cd build_cpp && \
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    ninja && \
    echo "C++ conceptual build complete."

# --- Stage 2: Final runtime image ---
FROM --platform=${TARGETPLATFORM:-ubuntu:22.04} ubuntu:22.04

ARG TARGETPLATFORM
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libstdc++6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
ENV APP_USER=aicompiler_user
RUN useradd --create-home --shell /bin/bash ${APP_USER}

WORKDIR /app

# Re-install Python dependencies using Poetry for the runtime stage
# Ensure same version as in builder
ENV POETRY_VERSION=1.7.1 
COPY pyproject.toml /app/
COPY poetry.lock /app/ 
RUN pip3 install --no-cache-dir poetry==${POETRY_VERSION} && \
    poetry install --no-interaction --no-ansi --all-extras --only main && \
    rm -rf /root/.cache/pypoetry

# Copy application code
COPY aicompiler /app/aicompiler/
COPY examples /app/examples/ 
# COPY --from=builder /app/build_cpp/libaicompiler_cpp_ops.so /app/aicompiler/lib/ # Example for C++ libs

RUN chown -R ${APP_USER}:${APP_USER} /app

USER ${APP_USER}

CMD ["python3", "examples/simple_example.py"] 