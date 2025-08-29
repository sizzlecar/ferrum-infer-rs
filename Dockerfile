# Ferrum Inference Engine Docker Image
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build the project
RUN cargo build --release

# Runtime stage
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built binary from builder
COPY --from=0 /app/target/release/ferrum-infer /usr/local/bin/

# Expose port
EXPOSE 8080

CMD ["ferrum-infer"]