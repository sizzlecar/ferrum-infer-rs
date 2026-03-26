//! GPU-side paged KV cache block pool.
//!
//! One large contiguous `CudaSlice` per layer per K/V, divided into
//! fixed-size blocks. Block allocation/deallocation is managed by the
//! CPU-side `BlockPool` in ferrum-kv; this struct owns the GPU memory.
//!
//! Block layout per pool: `[max_blocks, block_size, num_kv_heads, head_dim]`
//! Physical block `i` starts at offset `i * block_stride` elements.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

/// Configuration for the GPU paged KV block pool.
#[derive(Debug, Clone)]
pub struct GpuPagedKvConfig {
    pub block_size: usize,       // tokens per block (must be power of 2)
    pub max_blocks: usize,       // total physical blocks in pool
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

/// GPU-side paged KV cache block pool.
///
/// Manages two large buffers per layer (K and V) that hold all physical
/// blocks. The attention kernel indexes into these via a block table.
pub struct GpuPagedKvPool {
    /// k_pools[layer]: `[max_blocks * block_size * num_kv_heads * head_dim]`
    k_pools: Vec<CudaSlice<half::f16>>,
    /// v_pools[layer]: same layout
    v_pools: Vec<CudaSlice<half::f16>>,

    config: GpuPagedKvConfig,

    /// Elements per block: `block_size * num_kv_heads * head_dim`
    block_stride: usize,
    /// Elements per token within a block: `num_kv_heads * head_dim`
    token_stride: usize,

    stream: Arc<CudaStream>,
}

impl GpuPagedKvPool {
    /// Allocate the GPU block pool.
    pub fn new(
        config: GpuPagedKvConfig,
        stream: Arc<CudaStream>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        assert!(
            config.block_size.is_power_of_two(),
            "block_size must be power of 2, got {}",
            config.block_size
        );

        let token_stride = config.num_kv_heads * config.head_dim;
        let block_stride = config.block_size * token_stride;
        let pool_size = config.max_blocks * block_stride;

        let mut k_pools = Vec::with_capacity(config.num_layers);
        let mut v_pools = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            k_pools.push(unsafe { stream.alloc::<half::f16>(pool_size)? });
            v_pools.push(unsafe { stream.alloc::<half::f16>(pool_size)? });
        }

        tracing::info!(
            "GpuPagedKvPool allocated: {} layers × {} blocks × {} tok/block = {:.1}MB per K/V",
            config.num_layers,
            config.max_blocks,
            config.block_size,
            (pool_size * std::mem::size_of::<half::f16>()) as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            k_pools,
            v_pools,
            config,
            block_stride,
            token_stride,
            stream,
        })
    }

    /// Write one token's K data into the pool at the given physical block and slot.
    ///
    /// `k_data` must be `[num_kv_heads * head_dim]` elements (one token).
    pub fn write_k_token(
        &mut self,
        layer: usize,
        physical_block: usize,
        slot: usize,
        k_data: &cudarc::driver::CudaView<half::f16>,
    ) -> Result<(), cudarc::driver::DriverError> {
        let offset = physical_block * self.block_stride + slot * self.token_stride;
        let mut dst = self.k_pools[layer].slice_mut(offset..offset + self.token_stride);
        self.stream.memcpy_dtod(k_data, &mut dst)
    }

    /// Write one token's V data into the pool at the given physical block and slot.
    pub fn write_v_token(
        &mut self,
        layer: usize,
        physical_block: usize,
        slot: usize,
        v_data: &cudarc::driver::CudaView<half::f16>,
    ) -> Result<(), cudarc::driver::DriverError> {
        let offset = physical_block * self.block_stride + slot * self.token_stride;
        let mut dst = self.v_pools[layer].slice_mut(offset..offset + self.token_stride);
        self.stream.memcpy_dtod(v_data, &mut dst)
    }

    /// Bulk-copy contiguous KV from prefill into paged blocks.
    ///
    /// `k_contiguous` layout: `[seq_len, num_kv_heads, head_dim]`.
    /// `block_ids` maps logical block index → physical block ID.
    pub fn copy_contiguous_to_paged(
        &mut self,
        layer: usize,
        k_contiguous: &CudaSlice<half::f16>,
        v_contiguous: &CudaSlice<half::f16>,
        seq_len: usize,
        block_ids: &[i32],
    ) -> Result<(), cudarc::driver::DriverError> {
        let bs = self.config.block_size;
        for (logical_block, &physical_block) in block_ids.iter().enumerate() {
            let phys = physical_block as usize;
            let src_start = logical_block * bs * self.token_stride;
            let tokens_in_block = (seq_len - logical_block * bs).min(bs);
            let elems = tokens_in_block * self.token_stride;

            let k_src = k_contiguous.slice(src_start..src_start + elems);
            let dst_offset = phys * self.block_stride;
            let mut k_dst = self.k_pools[layer].slice_mut(dst_offset..dst_offset + elems);
            self.stream.memcpy_dtod(&k_src, &mut k_dst)?;

            let v_src = v_contiguous.slice(src_start..src_start + elems);
            let mut v_dst = self.v_pools[layer].slice_mut(dst_offset..dst_offset + elems);
            self.stream.memcpy_dtod(&v_src, &mut v_dst)?;
        }
        Ok(())
    }

    /// Upload a block table (logical → physical mapping) to GPU.
    pub fn upload_block_table(
        &self,
        block_ids: &[i32],
    ) -> Result<CudaSlice<i32>, cudarc::driver::DriverError> {
        self.stream.memcpy_stod(block_ids)
    }

    /// Get the K block pool buffer for a layer (for passing to kernel).
    pub fn k_pool(&self, layer: usize) -> &CudaSlice<half::f16> {
        &self.k_pools[layer]
    }

    /// Get the V block pool buffer for a layer.
    pub fn v_pool(&self, layer: usize) -> &CudaSlice<half::f16> {
        &self.v_pools[layer]
    }

    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    pub fn max_blocks(&self) -> usize {
        self.config.max_blocks
    }

    /// Total GPU memory used by all pool buffers (bytes).
    pub fn memory_bytes(&self) -> usize {
        let pool_elems = self.config.max_blocks * self.block_stride;
        pool_elems * std::mem::size_of::<half::f16>() * 2 * self.config.num_layers // K + V
    }
}
