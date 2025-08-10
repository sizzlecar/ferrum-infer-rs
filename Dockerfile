# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached unless Cargo.toml changes)
RUN cargo build --release && rm src/main.rs

# Copy source code
COPY src ./src

# Build the actual application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r ferrum && useradd -r -g ferrum ferrum

# Set working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/ferrum-infer .

# Change ownership
RUN chown ferrum:ferrum ferrum-infer

# Switch to non-root user
USER ferrum

# Set default environment variables
ENV FERRUM_INFER_HOST=0.0.0.0
ENV FERRUM_INFER_PORT=8080
ENV FERRUM_INFER_LOG_LEVEL=info
ENV FERRUM_INFER_CACHE_ENABLED=true
ENV FERRUM_INFER_CACHE_SIZE_MB=1024

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["./ferrum-infer"]