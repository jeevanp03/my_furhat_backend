# Use an official Python 3.11 runtime as a base image
FROM python:3.11-slim

# Install system dependencies, including build tools, git, cmake, clang, libc++-dev, libc++abi-dev, libomp-dev, ninja-build, Python development headers, OpenBLAS, and pkg-config
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    clang \
    libc++-dev \
    libc++abi-dev \
    libomp-dev \
    ninja-build \
    python3-dev \
    libopenblas-dev \
    pkg-config \
  && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel, then install Poetry 2.0.1
RUN pip install --upgrade pip setuptools wheel && \
    pip install poetry==2.0.1

# Set the working directory in the container
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock* /app/

# Install dependencies using Poetry without installing the root package
RUN poetry config virtualenvs.create false && \
    poetry install --no-root --no-interaction --no-ansi

# (Optional) Copy and install additional dependencies from requirements_poetry.txt if present
COPY requirements_poetry.txt /app/
RUN if [ -f requirements_poetry.txt ]; then pip install --no-cache-dir -r requirements_poetry.txt; fi

# # Set CMake arguments for OpenBLAS support
# ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

# # Install llama_cpp_python with verbose output
# RUN pip install --no-cache-dir --verbose llama_cpp_python==0.3.7

# RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install llama-cpp-python

RUN CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a" pip install llama-cpp-python


# Copy the rest of the project files
COPY . /app

# Expose the port if needed (or you can omit if not running the server automatically)
EXPOSE 8000

# Instead of running the app automatically, start a shell for interactive work
CMD [ "bash" ]
