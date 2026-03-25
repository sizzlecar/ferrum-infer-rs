//! CUDA Graph capture and replay for decode acceleration.
//!
//! Captures the entire decode forward pass as a CUDA Graph on the first call,
//! then replays it on subsequent calls — reducing ~550 kernel launches to 1.

use std::sync::Arc;

use candle_core::cuda_backend::cudarc;
use cudarc::driver::{CudaSlice, CudaStream};

/// Manages a captured CUDA Graph for decode replay.
pub struct CudaGraphState {
    graph: cudarc::driver::CudaGraph,
    stream: Arc<CudaStream>,
    pub token_id_buf: CudaSlice<u32>,
    pub position_buf: CudaSlice<u32>,
    pub valid_kv_len_buf: CudaSlice<u32>,
    uploaded: bool,
}

impl CudaGraphState {
    pub fn begin_capture(stream: &Arc<CudaStream>) -> candle_core::Result<()> {
        stream
            .begin_capture(cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| candle_core::Error::Msg(format!("Graph begin_capture failed: {e}")))
    }

    pub fn end_capture(
        stream: &Arc<CudaStream>,
    ) -> candle_core::Result<Option<cudarc::driver::CudaGraph>> {
        stream
            .end_capture(
                cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| candle_core::Error::Msg(format!("Graph end_capture failed: {e}")))
    }

    pub fn new(
        graph: cudarc::driver::CudaGraph,
        stream: Arc<CudaStream>,
    ) -> candle_core::Result<Self> {
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

    pub fn upload(&mut self) -> candle_core::Result<()> {
        if !self.uploaded {
            self.graph
                .upload()
                .map_err(|e| candle_core::Error::Msg(format!("Graph upload failed: {e}")))?;
            self.uploaded = true;
        }
        Ok(())
    }

    pub fn replay(
        &self,
        token_id: u32,
        position: u32,
        valid_kv_len: u32,
    ) -> candle_core::Result<()> {
        self.stream
            .memcpy_htod(&[token_id], &mut self.token_id_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("token_id update: {e}")))?;
        self.stream
            .memcpy_htod(&[position], &mut self.position_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("position update: {e}")))?;
        self.stream
            .memcpy_htod(&[valid_kv_len], &mut self.valid_kv_len_buf.clone())
            .map_err(|e| candle_core::Error::Msg(format!("valid_kv_len update: {e}")))?;

        self.graph
            .launch()
            .map_err(|e| candle_core::Error::Msg(format!("Graph launch failed: {e}")))
    }
}
