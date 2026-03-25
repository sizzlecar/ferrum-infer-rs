//! CUDA Graph capture and replay for decode acceleration.
//!
//! Captures the entire decode forward pass as a CUDA Graph on the first call,
//! then replays it on subsequent calls — reducing ~550 kernel launches to 1.
//!
//! Requirements for graph compatibility:
//! - All buffer addresses must be fixed (DecodeBuffers are pre-allocated)
//! - No dynamic shapes (decode is always batch=1, seq=1)
//! - No CPU-GPU synchronization during the captured pass
//! - KV cache uses full max_len buffer with valid_kv_len masking
//!
//! cudarc 0.19.4 auto-detects memory pool support and uses cudaMallocAsync
//! on modern GPUs, making stream-ordered allocations graph-compatible.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc;
use cudarc::driver::{CudaSlice, CudaStream};

/// Manages a captured CUDA Graph for decode replay.
///
/// The graph is captured from a single decode step and can be replayed
/// with updated parameters (token_id, position, valid_kv_len).
pub struct CudaGraphState {
    /// The captured and instantiated graph, ready for replay.
    graph: cudarc::driver::CudaGraph,
    /// Stream used for capture and replay.
    stream: Arc<CudaStream>,
    /// Device-side buffer for the input token ID (updated before each replay).
    /// The decode runner reads from this buffer during the captured pass.
    pub token_id_buf: CudaSlice<u32>,
    /// Device-side buffer for the current position (for RoPE indexing).
    pub position_buf: CudaSlice<u32>,
    /// Device-side buffer for valid KV length (for attention masking).
    pub valid_kv_len_buf: CudaSlice<u32>,
    /// Whether the graph has been uploaded (pre-upload optimization).
    uploaded: bool,
}

impl CudaGraphState {
    /// Begin capturing a CUDA Graph on the given stream.
    ///
    /// After calling this, all CUDA operations on the stream are recorded
    /// (not executed). Call `end_capture()` to finalize.
    pub fn begin_capture(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| candle_core::Error::Msg(format!("Graph begin_capture failed: {e}")))
    }

    /// End capture and create the executable graph.
    ///
    /// Returns `None` if no kernels were recorded (shouldn't happen for decode).
    pub fn end_capture(
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<Option<cudarc::driver::CudaGraph>> {
        stream
            .end_capture(0) // flags = 0 (default)
            .map_err(|e| candle_core::Error::Msg(format!("Graph end_capture failed: {e}")))
    }

    /// Create a CudaGraphState from a captured graph.
    pub fn new(
        graph: cudarc::driver::CudaGraph,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
        // Allocate small device-side parameter buffers
        let token_id_buf = unsafe {
            stream
                .alloc::<u32>(1)
                .map_err(|e| candle_core::Error::Msg(format!("token_id_buf alloc: {e}")))?
        };
        let position_buf = unsafe {
            stream
                .alloc::<u32>(1)
                .map_err(|e| candle_core::Error::Msg(format!("position_buf alloc: {e}")))?
        };
        let valid_kv_len_buf = unsafe {
            stream
                .alloc::<u32>(1)
                .map_err(|e| candle_core::Error::Msg(format!("valid_kv_len_buf alloc: {e}")))?
        };

        Ok(Self {
            graph,
            stream,
            token_id_buf,
            position_buf,
            valid_kv_len_buf,
            uploaded: false,
        })
    }

    /// Pre-upload graph resources to GPU for faster first launch.
    pub fn upload(&mut self) -> candle_core::Result<()> {
        if !self.uploaded {
            self.graph
                .upload()
                .map_err(|e| candle_core::Error::Msg(format!("Graph upload failed: {e}")))?;
            self.uploaded = true;
        }
        Ok(())
    }

    /// Update parameters and replay the captured graph.
    ///
    /// This is the hot path — a single GPU call replaces ~550 kernel launches.
    pub fn replay(
        &self,
        token_id: u32,
        position: u32,
        valid_kv_len: u32,
    ) -> candle_core::Result<()> {
        // Update device-side parameters via async memcpy
        self.stream
            .memcpy_htod(&[token_id], &mut self.token_id_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("token_id update: {e}")))?;
        self.stream
            .memcpy_htod(&[position], &mut self.position_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("position update: {e}")))?;
        self.stream
            .memcpy_htod(&[valid_kv_len], &mut self.valid_kv_len_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("valid_kv_len update: {e}")))?;

        // Launch the captured graph — single CPU→GPU call
        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("Graph launch failed: {e}")))
    }
}
