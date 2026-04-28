//! `GgufFile`: mmap-backed reader for a single GGUF file.
//!
//! Lifecycle:
//!   1. `GgufFile::open(path)` — mmaps the file and parses the header.
//!      No tensor payloads are read at this stage.
//!   2. `architecture()`, `metadata_*()`, `tensor_names()`, `tensor_info()` —
//!      cheap lookups, all served from the parsed header in memory.
//!   3. `read_tensor(name, device)` — slices the mmap at the right offset
//!      and asks candle to materialise a `QTensor` (still quantized).
//!
//! Tensor reads only need a shared `&self` because the mmap is immutable; the
//! file is safe to share across threads. (Candle's `Content::tensor` wants
//! a `&mut R: Read + Seek`, but we satisfy it with a fresh `Cursor<&[u8]>`
//! on each call — the cursor's mutable state is local to the call.)

use std::fs::File;
use std::io::Cursor;
use std::path::Path;

use candle_core::quantized::gguf_file::{Content, TensorInfo, Value};
use candle_core::quantized::QTensor;
use candle_core::{Device, Error as CandleError, Result as CandleResult};
use memmap2::Mmap;

/// Read-only handle to a memory-mapped GGUF file.
pub struct GgufFile {
    /// memory-mapped file payload. Kept alive for the lifetime of `self`
    /// because `read_tensor` slices into it.
    mmap: Mmap,
    /// Parsed header / metadata / tensor descriptors. No payload bytes.
    content: Content,
}

impl GgufFile {
    /// Open and parse the header of a GGUF file.
    ///
    /// Returns immediately after the descriptor table is read — no tensor
    /// data is materialised. `read_tensor` lazy-loads individual tensors.
    pub fn open(path: impl AsRef<Path>) -> CandleResult<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(|e| {
            CandleError::Msg(format!(
                "failed to open GGUF file '{}': {e}",
                path_ref.display()
            ))
        })?;
        // SAFETY: `Mmap::map` requires that the underlying file is not modified
        // while the mapping is live. We treat the file as read-only for the
        // entire lifetime of `self`. `Mmap` itself only exposes `&[u8]`.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
            CandleError::Msg(format!(
                "failed to mmap GGUF file '{}': {e}",
                path_ref.display()
            ))
        })?;
        let mut cursor = Cursor::new(&mmap[..]);
        let content = Content::read(&mut cursor)?;
        Ok(Self { mmap, content })
    }

    /// Raw access to candle's parsed header — for callers that need the full
    /// `metadata` / `tensor_infos` maps. Prefer the typed accessors below.
    pub fn content(&self) -> &Content {
        &self.content
    }

    // ── Metadata: typed accessors ─────────────────────────────────────────
    //
    // GGUF metadata keys are conventionally `<scope>.<field>` strings, e.g.
    // `general.architecture` or `qwen3.block_count`. Different model families
    // namespace under their architecture id. `architecture()` is the one key
    // that's always present and tells you which scope to read the rest from.

    /// Architecture string, e.g. `"qwen3"`, `"llama"`. Read from
    /// `general.architecture`. Errors if the key is missing or non-string.
    pub fn architecture(&self) -> CandleResult<&str> {
        self.metadata_string("general.architecture")
    }

    /// Raw metadata value lookup. Returns `None` if the key is absent.
    pub fn metadata(&self, key: &str) -> Option<&Value> {
        self.content.metadata.get(key)
    }

    /// Read a string-typed metadata field. Errors if missing or wrong type.
    pub fn metadata_string(&self, key: &str) -> CandleResult<&str> {
        self.require_metadata(key)?.to_string().map(|s| s.as_str())
    }

    /// Read a u32-typed metadata field. Errors if missing or wrong type.
    pub fn metadata_u32(&self, key: &str) -> CandleResult<u32> {
        self.require_metadata(key)?.to_u32()
    }

    /// Read a u64-typed metadata field. Errors if missing or wrong type.
    pub fn metadata_u64(&self, key: &str) -> CandleResult<u64> {
        self.require_metadata(key)?.to_u64()
    }

    /// Read an f32-typed metadata field. Errors if missing or wrong type.
    pub fn metadata_f32(&self, key: &str) -> CandleResult<f32> {
        self.require_metadata(key)?.to_f32()
    }

    /// Read a bool-typed metadata field. Errors if missing or wrong type.
    pub fn metadata_bool(&self, key: &str) -> CandleResult<bool> {
        self.require_metadata(key)?.to_bool()
    }

    fn require_metadata(&self, key: &str) -> CandleResult<&Value> {
        self.metadata(key)
            .ok_or_else(|| CandleError::Msg(format!("GGUF metadata key missing: '{key}'")))
    }

    // ── Tensor enumeration ────────────────────────────────────────────────

    /// Total number of tensors declared in the header.
    pub fn tensor_count(&self) -> usize {
        self.content.tensor_infos.len()
    }

    /// Iterate over every tensor name in the file. Order is whatever the
    /// underlying `HashMap` yields — do not rely on it being deterministic.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.content.tensor_infos.keys().map(|s| s.as_str())
    }

    /// Look up a tensor descriptor (shape, dtype, byte offset) without
    /// touching the payload. `None` if the tensor isn't in the file.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.content.tensor_infos.get(name)
    }

    /// Whether a tensor with `name` is declared in the header.
    pub fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    // ── Tensor read ───────────────────────────────────────────────────────

    /// Materialise a tensor as a candle `QTensor` on the target device.
    ///
    /// The returned tensor is **still quantized** — no dequant happens here.
    /// Wrap it in `QMatMul::from_qtensor` for inference, or call
    /// `QTensor::dequantize(device)` to get a fp32 `Tensor`.
    pub fn read_tensor(&self, name: &str, device: &Device) -> CandleResult<QTensor> {
        let mut cursor = Cursor::new(&self.mmap[..]);
        self.content.tensor(&mut cursor, name, device)
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufFile")
            .field("size_bytes", &self.mmap.len())
            .field("metadata_keys", &self.content.metadata.len())
            .field("tensor_count", &self.content.tensor_infos.len())
            .field("tensor_data_offset", &self.content.tensor_data_offset)
            .finish()
    }
}
