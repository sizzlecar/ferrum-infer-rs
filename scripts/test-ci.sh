#!/bin/bash

# CI Test Script
# This script simulates the CI environment and tests the build process

set -e

echo "🚀 Starting CI test simulation..."

# Set CI environment variable
export CI=true

# Install system dependencies (simulate CI environment)
echo "📦 Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y libssl-dev pkg-config build-essential
elif command -v yum &> /dev/null; then
    sudo yum install -y openssl-devel pkg-config gcc
elif command -v brew &> /dev/null; then
    brew install openssl pkg-config
else
    echo "⚠️  Unknown package manager. Please install openssl-dev and pkg-config manually."
fi

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "🦀 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Add rustfmt and clippy
echo "🔧 Installing Rust components..."
rustup component add rustfmt clippy

echo "✨ Formatting check..."
cargo fmt --all -- --check

echo "🔍 Clippy check (warnings allowed)..."
cargo clippy --lib -- -A warnings

echo "🏗️  Basic compilation check..."
cargo check --lib

echo "📚 Building library..."
cargo build --lib

echo "🧪 Running tests..."
cargo test --lib --no-fail-fast

echo "✅ All CI checks passed!"

echo "
📊 CI Test Summary:
- ✅ Code formatting (rustfmt)
- ✅ Linting (clippy) 
- ✅ Compilation check
- ✅ Library build
- ✅ Unit tests

🎉 Your code is ready for CI/CD!"