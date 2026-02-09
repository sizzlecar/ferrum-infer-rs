//! Metal GPU-side sampling (top-k / top-p / temperature / repetition penalty).
//!
//! This module is designed to avoid transferring full-vocab logits back to CPU every token.
//! It works by:
//! - extracting the underlying Metal buffer from a Candle Metal tensor
//! - selecting Top-K candidates via repeated argmax + in-place masking on GPU (correct, simple)
//! - running a small-k sampling kernel on GPU (temperature + top-p + repetition penalty)
//! - reading back only the sampled token id (u32)
//!
//! Notes:
//! - This path is currently optimized for correctness and minimal integration risk.
//! - K should be small (e.g. 32/64). Larger K increases the number of argmax passes.

use crate::metal::{MetalContext, MetalError};
use ferrum_types::FerrumError;

use candle_core::{DType, IndexOp, Tensor, D};
use metal::{Buffer, ComputePipelineState, MTLSize};
use std::sync::Arc;

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use candle_core::{Layout, Storage};

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
use candle_core::MetalStorage;

/// GPU sampling ops backed by Ferrum's embedded metallib.
pub struct MetalSamplingOps {
    context: Arc<MetalContext>,
    mask_one: ComputePipelineState,
    sample_from_topk: ComputePipelineState,
}

impl MetalSamplingOps {
    pub fn new(context: Arc<MetalContext>) -> Result<Self, FerrumError> {
        let library = context
            .library()
            .ok_or_else(|| MetalError::generic("Metal library not loaded"))?;

        let mask_one_fn = library
            .get_function("sampling_mask_one", None)
            .map_err(|e| {
                MetalError::compilation_failed(format!("Missing sampling_mask_one: {}", e))
            })?;
        let sample_fn = library
            .get_function("sampling_sample_from_topk", None)
            .map_err(|e| {
                MetalError::compilation_failed(format!("Missing sampling_sample_from_topk: {}", e))
            })?;

        let mask_one = context
            .device
            .new_compute_pipeline_state_with_function(&mask_one_fn)
            .map_err(|e| MetalError::compilation_failed(format!("mask_one pipeline: {}", e)))?;
        let sample_from_topk = context
            .device
            .new_compute_pipeline_state_with_function(&sample_fn)
            .map_err(|e| MetalError::compilation_failed(format!("sample pipeline: {}", e)))?;

        Ok(Self {
            context,
            mask_one,
            sample_from_topk,
        })
    }

    /// Attempt to sample a token on GPU from a logits tensor on Metal.
    ///
    /// - `logits`: logits tensor (any shape where last dim is vocab)
    /// - `top_k`: number of candidates to consider
    /// - `top_p`: nucleus threshold (0..1)
    /// - `temperature`: temperature (>0)
    /// - `repetition_penalty`: repetition penalty (>=1 typically)
    /// - `rep_token_ids` / `rep_token_freqs`: sparse repetition list
    /// - `rng_seed`: per-step random seed
    pub fn sample_token(
        &self,
        logits: &Tensor,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        rep_token_ids: &[u32],
        rep_token_freqs: &[u32],
        rng_seed: u32,
    ) -> Result<u32, FerrumError> {
        // 1) Normalize logits shape to 1D [vocab] on Metal
        let logits_1d = match logits.dims().len() {
            1 => logits.clone(),
            2 => logits
                .i(0)
                .map_err(|e| FerrumError::internal(format!("Index batch failed: {}", e)))?,
            3 => {
                let seq_len = logits.dims()[1];
                logits
                    .i((0, seq_len.saturating_sub(1)))
                    .map_err(|e| FerrumError::internal(format!("Index last token failed: {}", e)))?
            }
            _ => logits
                .flatten_all()
                .map_err(|e| FerrumError::internal(format!("Flatten failed: {}", e)))?,
        };

        // Ensure float logits for our kernels.
        let logits_1d = if logits_1d.dtype() != DType::F32 {
            logits_1d
                .to_dtype(DType::F32)
                .map_err(|e| FerrumError::internal(format!("Cast logits to f32 failed: {}", e)))?
        } else {
            logits_1d
        };

        // 2) Extract underlying Metal buffer + offset
        let (metal_storage, layout) = extract_metal_storage_and_layout(&logits_1d)?;
        if !layout.is_contiguous() {
            return Err(FerrumError::internal(
                "GPU sampling requires contiguous logits layout",
            ));
        }
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| FerrumError::internal("Non-contiguous logits layout"))?;
        let vocab_size = (end - start) as u32;

        let logits_buf = metal_storage.buffer();
        let logits_offset_bytes = (start * std::mem::size_of::<f32>()) as u64;

        // 3) Select Top-K indices using repeated argmax + in-place mask.
        let k = top_k.clamp(1, 256);
        let mut topk_indices: Vec<u32> = Vec::with_capacity(k);

        for _ in 0..k {
            let idx = logits_1d
                .argmax(D::Minus1)
                .map_err(|e| FerrumError::internal(format!("Argmax failed: {}", e)))?
                .to_device(&candle_core::Device::Cpu)
                .map_err(|e| FerrumError::internal(format!("Argmax to CPU failed: {}", e)))?
                .to_vec0::<u32>()
                .map_err(|e| FerrumError::internal(format!("Argmax readback failed: {}", e)))?;

            topk_indices.push(idx);

            // Mask this index in-place on GPU so next argmax finds next candidate.
            self.dispatch_mask_one(logits_buf, logits_offset_bytes, vocab_size, idx)?;
        }

        // 4) Run sampling kernel over the K candidates.
        let topk_buf = self.create_u32_buffer(&topk_indices, "topk_indices")?;
        let rep_ids_buf = self.create_u32_buffer(rep_token_ids, "rep_token_ids")?;
        let rep_freqs_buf = self.create_u32_buffer(rep_token_freqs, "rep_token_freqs")?;
        let out_buf = self.context.device.new_buffer(
            std::mem::size_of::<u32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        out_buf.set_label("sample_out");

        self.dispatch_sample_from_topk(
            logits_buf,
            logits_offset_bytes,
            &topk_buf,
            &rep_ids_buf,
            &rep_freqs_buf,
            &out_buf,
            vocab_size,
            k as u32,
            rep_token_ids.len() as u32,
            temperature,
            top_p,
            repetition_penalty,
            rng_seed,
        )?;

        // Read back sampled token id
        let out_ptr = out_buf.contents() as *const u32;
        let token = unsafe { *out_ptr };
        Ok(token)
    }

    fn dispatch_mask_one(
        &self,
        logits_buf: &Buffer,
        logits_offset: u64,
        vocab_size: u32,
        token_id: u32,
    ) -> Result<(), FerrumError> {
        let vocab_size_buf = self.create_scalar_u32(vocab_size, "vocab_size")?;
        let token_id_buf = self.create_scalar_u32(token_id, "mask_token_id")?;

        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mask_one);
        encoder.set_buffer(0, Some(logits_buf), logits_offset);
        encoder.set_buffer(1, Some(&vocab_size_buf), 0);
        encoder.set_buffer(2, Some(&token_id_buf), 0);

        // Single threadgroup is enough (kernel checks tid==0).
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_sample_from_topk(
        &self,
        logits_buf: &Buffer,
        logits_offset: u64,
        topk_indices: &Buffer,
        rep_ids: &Buffer,
        rep_freqs: &Buffer,
        output: &Buffer,
        vocab_size: u32,
        k: u32,
        rep_len: u32,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        rng_seed: u32,
    ) -> Result<(), FerrumError> {
        let vocab_size_buf = self.create_scalar_u32(vocab_size, "vocab_size")?;
        let k_buf = self.create_scalar_u32(k, "k")?;
        let rep_len_buf = self.create_scalar_u32(rep_len, "rep_len")?;
        let temperature_buf = self.create_scalar_f32(temperature, "temperature")?;
        let top_p_buf = self.create_scalar_f32(top_p, "top_p")?;
        let rep_pen_buf = self.create_scalar_f32(repetition_penalty, "repetition_penalty")?;
        let seed_buf = self.create_scalar_u32(rng_seed, "rng_seed")?;

        let command_buffer = self.context.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.sample_from_topk);

        encoder.set_buffer(0, Some(logits_buf), logits_offset);
        encoder.set_buffer(1, Some(topk_indices), 0);
        encoder.set_buffer(2, Some(rep_ids), 0);
        encoder.set_buffer(3, Some(rep_freqs), 0);
        encoder.set_buffer(4, Some(output), 0);
        encoder.set_buffer(5, Some(&vocab_size_buf), 0);
        encoder.set_buffer(6, Some(&k_buf), 0);
        encoder.set_buffer(7, Some(&rep_len_buf), 0);
        encoder.set_buffer(8, Some(&temperature_buf), 0);
        encoder.set_buffer(9, Some(&top_p_buf), 0);
        encoder.set_buffer(10, Some(&rep_pen_buf), 0);
        encoder.set_buffer(11, Some(&seed_buf), 0);

        // One threadgroup of 256 threads is enough for K<=256.
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(())
    }

    fn create_u32_buffer(&self, data: &[u32], label: &str) -> Result<Buffer, FerrumError> {
        let bytes: &[u8] = bytemuck::cast_slice::<u32, u8>(data);
        let buffer = self.context.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        Ok(buffer)
    }

    fn create_scalar_u32(&self, v: u32, label: &str) -> Result<Buffer, FerrumError> {
        self.create_u32_buffer(&[v], label)
    }

    fn create_scalar_f32(&self, v: f32, label: &str) -> Result<Buffer, FerrumError> {
        let binding = [v];
        let bytes: &[u8] = bytemuck::cast_slice::<f32, u8>(&binding);
        let buffer = self.context.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        buffer.set_label(label);
        Ok(buffer)
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
fn extract_metal_storage_and_layout(t: &Tensor) -> Result<(MetalStorage, Layout), FerrumError> {
    let (storage_guard, layout) = t.storage_and_layout();
    let layout = layout.clone();

    match &*storage_guard {
        Storage::Metal(ms) => Ok((ms.clone(), layout)),
        _ => Err(FerrumError::internal(
            "Expected Metal storage for GPU sampling",
        )),
    }
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
fn extract_metal_storage_and_layout(_t: &Tensor) -> Result<((), ()), FerrumError> {
    Err(FerrumError::unsupported(
        "Metal sampling not available on this platform",
    ))
}
