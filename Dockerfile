# CPU-only Ferrum image. Portable — runs on any x86_64 Linux with Docker.
#
# Build:
#   docker build -t ferrum:cpu .
#
# Run (chat):
#   docker run --rm -it \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     ferrum:cpu run qwen3:0.6b
#
# Run (HTTP server):
#   docker run --rm -p 8000:8000 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     ferrum:cpu serve --model qwen3:0.6b --port 8000
#
# Hugging Face auth (for gated models):
#   docker run --rm -it \
#     -e HF_TOKEN=hf_xxx \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     ferrum:cpu pull meta-llama/Llama-3.2-1B-Instruct

# ── Build stage ──────────────────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock rustfmt.toml ./
COPY crates ./crates

RUN cargo build --release -p ferrum-cli --bin ferrum

# ── Runtime stage ────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/ferrum /usr/local/bin/ferrum

# Default HF cache (override with -v host/path:/root/.cache/huggingface)
ENV HF_HOME=/root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

EXPOSE 8000

ENTRYPOINT ["ferrum"]
CMD ["--help"]
