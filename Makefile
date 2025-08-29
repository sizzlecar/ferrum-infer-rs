.PHONY: help build build-cpu build-docker test clean run

help:
	@echo "Ferrum Inference Engine - Build Commands"
	@echo ""
	@echo "  make build        - Build with default features"
	@echo "  make build-cpu    - Build CPU-only version (no CUDA)"
	@echo "  make build-docker - Build in Docker container"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make run          - Run the server"
	@echo ""

# Default build (may fail on macOS due to Candle issues)
build:
	cargo build --release

# CPU-only build (safer for macOS)
build-cpu:
	cargo build --release --no-default-features

# Build in Docker (recommended for production)
build-docker:
	docker build -t ferrum-infer:latest .

# Run tests
test:
	cargo test --all

# Clean build artifacts
clean:
	cargo clean
	rm -rf target/

# Run the server
run:
	cargo run --release

# Development build (faster compilation)
dev:
	cargo build

# Format code
fmt:
	cargo fmt --all

# Check code
check:
	cargo check --all
	cargo clippy --all

# Install for Linux with CUDA
install-cuda-deps:
	@echo "Installing CUDA dependencies..."
	@echo "Please ensure CUDA 12.3+ is installed"
	@echo "Visit: https://developer.nvidia.com/cuda-downloads"

# Setup for macOS
setup-macos:
	@echo "Setting up for macOS..."
	@echo "Note: Full GPU support is limited on macOS"
	@echo "Consider using Docker for production deployments"
	brew install libomp
	
# Setup for Ubuntu
setup-ubuntu:
	sudo apt-get update
	sudo apt-get install -y build-essential pkg-config libssl-dev
	@echo "For GPU support, install CUDA 12.3+"
