//! GGUF (GGML Universal Format) reader.
//!
//! Phase 1A scope: parse the file header, expose metadata + tensor descriptors,
//! and load individual tensors as candle `QTensor` (which already handles
//! dequant for every K-quant variant on CPU / Metal / CUDA).
//!
//! Out of scope here (lands in 1B / 1C): mapping the GGUF tensor names
//! (`blk.0.attn_q.weight`) to ferrum's model-config naming, wrapping `QTensor`
//! into the project's `Linear<B>` trait, and implementing `WeightLoader`.
//!
//! ## Why wrap candle instead of writing a parser from scratch
//!
//! `candle_core::quantized::gguf_file::Content` already implements the full
//! GGUF v1/v2/v3 spec, including all current GGML K-quant variants and
//! Metal/CUDA/CPU dequant kernels. Re-implementing that for ferrum would be
//! 3-5 weeks of work duplicating well-tested code. Instead this module
//! provides a small adapter that:
//!
//!   1. Adds an `mmap`-backed `open(path)` constructor (candle's API takes
//!      a generic `Read + Seek` and pushes file handling to the caller).
//!   2. Provides typed metadata accessors keyed by string (`metadata_string`,
//!      `metadata_u32`, …) so callers don't pattern-match on `Value` everywhere.
//!   3. Documents the GGUF metadata key conventions ferrum relies on
//!      (`general.architecture`, `<arch>.block_count`, …) in one place.

pub mod file;

pub use file::GgufFile;

// Re-exports — callers can import these from `ferrum_quantization::gguf` rather
// than reaching into `candle_core::quantized::*` directly. Keeps the dep
// surface explicit and lets us swap in a native parser later without churning
// downstream call sites.
pub use candle_core::quantized::gguf_file::{TensorInfo, Value, ValueType};
pub use candle_core::quantized::{GgmlDType, QTensor};
