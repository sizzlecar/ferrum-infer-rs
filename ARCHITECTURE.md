# LLM Inference Engine Architecture

This document describes the architecture of the Rust-based LLM inference engine MVP, focusing on performance, extensibility, and OpenAI API compatibility.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Module Design](#module-design)
5. [Data Flow](#data-flow)
6. [Performance Optimizations](#performance-optimizations)
7. [Extensibility](#extensibility)
8. [API Compatibility](#api-compatibility)
9. [Configuration](#configuration)
10. [Error Handling](#error-handling)

## Overview

The LLM Inference Engine is a high-performance, single-node inference server designed as an MVP to validate Rust's performance advantages for LLM serving. It provides OpenAI-compatible REST API endpoints while implementing efficient caching and memory management.

### Key Design Goals

- **Performance**: Maximize throughput and minimize latency
- **Compatibility**: Full OpenAI API compatibility
- **Extensibility**: Modular design for future distributed capabilities
- **Reliability**: Robust error handling and graceful degradation
- **Observability**: Comprehensive metrics and logging

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   HTTP Client   │    │   HTTP Client   │    │   HTTP Client   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                      ┌─────────┴───────────┐
                      │    Load Balancer    │  (Future)
                      │    (Not in MVP)     │
                      └─────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
    ┌─────────▼──────────┐                       │
    │                    │                       │
    │   HTTP Server      │                       │
    │   (Actix Web)      │                       │
    │                    │                       │
    │  ┌──────────────┐  │                       │
    │  │ Middleware   │  │                       │
    │  │ - Auth       │  │                       │
    │  │ - Logging    │  │                       │
    │  │ - CORS       │  │                       │
    │  └──────────────┘  │                       │
    │                    │                       │
    │  ┌──────────────┐  │                       │
    │  │   Handlers   │  │                       │
    │  │ - Chat       │  │                       │
    │  │ - Completion │  │                       │
    │  │ - Models     │  │                       │
    │  │ - Health     │  │                       │
    │  └──────────────┘  │                       │
    └─────────┬──────────┘                       │
              │                                  │
    ┌─────────▼──────────┐                       │
    │                    │                       │
    │ Inference Engine   │                       │
    │                    │                       │
    │ ┌────────────────┐ │     ┌───────────────┐ │
    │ │ Request Queue  │ │     │  KV Cache     │ │
    │ │ (Future)       │ │     │  - LRU        │ │
    │ └────────────────┘ │     │  - LFU        │ │
    │                    │     │  - FIFO       │ │
    │ ┌────────────────┐ │     └───────────────┘ │
    │ │ Model Manager  │ │                       │
    │ └────────────────┘ │                       │
    │                    │                       │
    │ ┌────────────────┐ │                       │
    │ │ Metrics        │ │                       │
    │ │ Collector      │ │                       │
    │ └────────────────┘ │                       │
    └─────────┬──────────┘                       │
              │                                  │
    ┌─────────▼──────────┐                       │
    │                    │                       │
    │   Model Layer      │                       │
    │                    │                       │
    │ ┌────────────────┐ │                       │
    │ │   Candle       │ │                       │
    │ │   Framework    │ │                       │
    │ └────────────────┘ │                       │
    │                    │                       │
    │ ┌────────────────┐ │                       │
    │ │  Tokenizer     │ │                       │
    │ └────────────────┘ │                       │
    └────────────────────┘                       │
                                                 │
              Hardware Layer                     │
    ┌─────────────────────────────────────────┐  │
    │  CPU / GPU / Memory / Storage           │  │
    └─────────────────────────────────────────┘  │
                                                 │
                 Future Extensions               │
    ┌─────────────────────────────────────────┐  │
    │  - Distributed Model Sharding          │  │
    │  - Dynamic Batching                     │  │
    │  - Multi-Node Coordination              │  │
    │  - Advanced Scheduling                  │  │
    └─────────────────────────────────────────┘  │
                                                 │
└─────────────────────────────────────────────────┘
```

## Core Components

### 1. HTTP Server Layer (`api` module)

**Responsibility**: Handle HTTP requests and provide OpenAI-compatible endpoints.

- **Framework**: Actix Web for high-performance async HTTP serving
- **Middleware**: Authentication, CORS, request logging, rate limiting
- **Endpoints**: 
  - `/v1/chat/completions` - Chat completion (streaming & non-streaming)
  - `/v1/completions` - Text completion (legacy)
  - `/v1/models` - List available models
  - `/health` - Health check
  - `/metrics` - Prometheus metrics

### 2. Inference Engine (`inference` module)

**Responsibility**: Orchestrate model loading, request processing, and response generation.

**Key Features**:
- Asynchronous request processing
- Connection pooling and resource management
- Request validation and preprocessing
- Response formatting and streaming
- Performance metrics collection

### 3. Model Management (`models` module)

**Responsibility**: Load, manage, and provide access to LLM models.

**Design Patterns**:
- **Trait-based abstraction**: `Model` trait for different model types
- **Factory pattern**: `ModelLoader` for different model sources
- **Singleton pattern**: `ModelManager` for model lifecycle

### 4. Caching System (`cache` module)

**Responsibility**: Implement efficient KV caching for inference acceleration.

**Cache Implementations**:
- **LRU Cache**: Least Recently Used eviction
- **LFU Cache**: Least Frequently Used eviction  
- **FIFO Cache**: First In, First Out eviction

**Features**:
- Memory-bounded caching
- TTL-based expiration
- Thread-safe concurrent access
- Cache hit/miss metrics

### 5. Configuration Management (`config` module)

**Responsibility**: Centralized configuration with environment variable support.

**Configuration Categories**:
- Server settings (host, port, timeouts)
- Model configuration (path, device, parameters)
- Cache settings (size, eviction policy)
- Performance tuning (batch size, threading)
- Monitoring (logging, metrics)

### 6. Error Handling (`error` module)

**Responsibility**: Structured error handling with HTTP response mapping.

**Error Categories**:
- Configuration errors
- Model loading errors
- Inference errors
- Cache errors
- Resource errors
- Validation errors

## Module Design

### Module Dependencies

```
main.rs
├── lib.rs
│   ├── api/
│   │   ├── handlers.rs
│   │   ├── routes.rs
│   │   ├── types.rs
│   │   └── middleware/
│   │       ├── auth.rs
│   │       └── logging.rs
│   ├── inference.rs
│   ├── models.rs
│   ├── cache.rs
│   ├── config.rs
│   ├── error.rs
│   ├── metrics.rs
│   └── utils.rs
```

### Trait Design

```rust
// Core abstraction for LLM models
#[async_trait]
pub trait Model: Send + Sync {
    fn model_info(&self) -> ModelInfo;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String>;
    async fn generate(
        &self,
        input_tokens: &[u32],
        generation_config: &GenerationConfig,
        cache: Option<&mut dyn InferenceCache>,
    ) -> Result<GenerationResult>;
    // ... additional methods
}

// Abstraction for different cache implementations
pub trait InferenceCache: Send + Sync {
    fn get_cache(&self, sequence_id: &str) -> Option<CacheEntry>;
    fn store_cache(&mut self, sequence_id: &str, cache: CacheEntry);
    fn remove_cache(&mut self, sequence_id: &str);
    fn clear_cache(&mut self);
    fn cache_stats(&self) -> CacheStats;
}

// Factory for loading different model types
#[async_trait]
pub trait ModelLoader: Send + Sync {
    async fn load_model(&self, config: &ModelConfig) -> Result<Box<dyn Model>>;
    async fn validate_model(&self, config: &ModelConfig) -> Result<()>;
    fn supported_model_types(&self) -> Vec<String>;
}
```

## Data Flow

### Request Processing Flow

```
1. HTTP Request Received
   ↓
2. Middleware Processing
   ├── Authentication Check
   ├── Request Logging
   └── CORS Headers
   ↓
3. Request Validation
   ├── Parameter Validation
   ├── Model Name Check
   └── Token Limits
   ↓
4. Cache Lookup
   ├── Generate Cache Key
   ├── Check KV Cache
   └── Return if Cache Hit
   ↓
5. Model Inference
   ├── Tokenize Input
   ├── Load/Get Model
   ├── Generate Response
   └── Update Cache
   ↓
6. Response Processing
   ├── Format Response
   ├── Stream if Requested
   └── Update Metrics
   ↓
7. HTTP Response Sent
```

### Streaming Flow

```
1. Streaming Request
   ↓
2. Create Response Channel
   ↓
3. Spawn Background Task
   ├── Process Request
   ├── Generate Chunks
   └── Send via Channel
   ↓
4. Stream HTTP Response
   ├── Server-Sent Events
   ├── JSON Chunks
   └── [DONE] Marker
```

## Performance Optimizations

### Memory Management

- **Zero-copy operations** where possible
- **Memory pools** for frequent allocations
- **Reference counting** with `Arc<T>` for shared data
- **Read-write locks** (`RwLock`) for concurrent access

### Concurrency

- **Async/await** for non-blocking I/O
- **Thread pools** for CPU-intensive tasks
- **Lock-free data structures** where applicable
- **Parallel request processing**

### Caching Strategy

- **Multi-level caching**:
  - L1: In-memory KV cache
  - L2: Model weight caching
  - L3: Tokenizer cache
- **Intelligent eviction** based on access patterns
- **Cache warming** for frequently used models

### Resource Optimization

- **Memory mapping** for large model files
- **Batch processing** for multiple requests
- **Connection pooling** for external services
- **Lazy loading** of model components

## Extensibility

### Future Distributed Architecture

The current MVP design enables easy extension to distributed deployment:

```rust
// Future distributed components
pub trait NodeManager {
    async fn register_node(&self, node_info: NodeInfo) -> Result<()>;
    async fn discover_nodes(&self) -> Result<Vec<NodeInfo>>;
    async fn health_check_nodes(&self) -> Result<Vec<NodeHealth>>;
}

pub trait ModelShard {
    async fn load_shard(&self, shard_config: ShardConfig) -> Result<()>;
    async fn forward(&self, input: ShardInput) -> Result<ShardOutput>;
}

pub trait RequestRouter {
    async fn route_request(&self, request: InferenceRequest) -> Result<NodeId>;
    fn update_routing_table(&self, routing_info: RoutingTable);
}
```

### Plugin Architecture

```rust
// Plugin system for extensibility
pub trait InferencePlugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    async fn initialize(&self, config: PluginConfig) -> Result<()>;
    async fn pre_inference(&self, request: &mut InferenceRequest) -> Result<()>;
    async fn post_inference(&self, response: &mut InferenceResponse) -> Result<()>;
}
```

## API Compatibility

### OpenAI API Coverage

| Endpoint | Status | Notes |
|----------|--------|-------|
| `/v1/chat/completions` | ✅ Complete | Full streaming support |
| `/v1/completions` | ✅ Complete | Legacy endpoint |
| `/v1/models` | ✅ Complete | Lists available models |
| `/v1/engines` | ✅ Deprecated | Redirects to models |
| `/v1/embeddings` | 🔄 Future | Not in MVP |
| `/v1/fine-tuning` | 🔄 Future | Not in MVP |

### Request/Response Compatibility

The engine maintains full compatibility with OpenAI's request and response formats:

```rust
// OpenAI Chat Completion Request
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}

// OpenAI Chat Completion Response
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

## Configuration

### Environment Variables

```bash
# Server Configuration
LLM_ENGINE_HOST=0.0.0.0
LLM_ENGINE_PORT=8080
LLM_ENGINE_API_KEY=your-secret-key

# Model Configuration
LLM_ENGINE_MODEL_PATH=microsoft/DialoGPT-medium
LLM_ENGINE_DEVICE=cpu
LLM_ENGINE_MAX_SEQUENCE_LENGTH=2048

# Cache Configuration
LLM_ENGINE_CACHE_ENABLED=true
LLM_ENGINE_CACHE_SIZE_MB=1024

# Logging Configuration
LLM_ENGINE_LOG_LEVEL=info
```

### Configuration File (TOML)

```toml
[server]
host = "0.0.0.0"
port = 8080
max_concurrent_requests = 100
enable_cors = true

[model]
name = "microsoft/DialoGPT-medium"
model_path = "microsoft/DialoGPT-medium"
device = "cpu"
max_sequence_length = 2048

[cache]
enabled = true
max_size_mb = 1024
eviction_policy = "lru"
max_sequences = 1000

[performance]
batch_size = 1
memory_pool_size_mb = 512
use_mmap = true

[logging]
level = "info"
format = "pretty"
log_requests = true

[metrics]
enabled = true
endpoint = "/metrics"
```

## Error Handling

### Error Hierarchy

```rust
pub enum EngineError {
    Config { message: String },           // Configuration errors
    Model { message: String },            // Model loading/inference errors
    Inference { message: String },        // Inference computation errors
    Cache { message: String },            // Cache operation errors
    InvalidRequest { message: String },   // Request validation errors
    Resource { message: String },         // Resource management errors
    Internal { message: String },         // Internal server errors
    Io(std::io::Error),                  // I/O errors
    Serde(serde_json::Error),            // Serialization errors
    Candle(candle_core::Error),          // ML framework errors
    Tokenizer(tokenizers::Error),        // Tokenizer errors
}
```

### HTTP Error Mapping

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| `InvalidRequest` | 400 Bad Request | Invalid parameters |
| `Model` | 503 Service Unavailable | Model not available |
| `Resource` | 507 Insufficient Storage | Out of memory/disk |
| `Internal` | 500 Internal Server Error | Unexpected errors |
| `Config` | 500 Internal Server Error | Configuration issues |

### Error Response Format

```json
{
  "error": {
    "message": "Invalid request: temperature must be between 0.0 and 2.0",
    "type": "invalid_request_error",
    "code": "INVALID_REQUEST"
  }
}
```

## Monitoring and Observability

### Metrics

- **Request metrics**: Total requests, success/failure rates, latency percentiles
- **Model metrics**: Inference time, tokens per second, model load time
- **Cache metrics**: Hit rate, miss rate, eviction count, memory usage
- **System metrics**: CPU usage, memory usage, disk I/O, network I/O

### Logging

- **Structured logging** with JSON format support
- **Request tracing** with unique request IDs
- **Performance logging** for slow requests
- **Error logging** with full context

### Health Checks

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "total_requests": 1234,
  "cache_hit_rate": 0.85,
  "memory_usage_mb": 2048
}
```

This architecture provides a solid foundation for the MVP while maintaining extensibility for future distributed deployment and advanced features.