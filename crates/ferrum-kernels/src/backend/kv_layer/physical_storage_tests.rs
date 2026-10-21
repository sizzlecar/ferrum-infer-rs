use super::*;
use ferrum_types::KvCacheStorage;

#[test]
fn physical_kv_storage_cpu_default_marker_matches_allocated_f32_payload() {
    use crate::backend::cpu::CpuBackend;
    let mut ctx = CpuBackend::new_context(&Default::default()).unwrap();
    let (k, v) = <KvFp16 as KvLayer<CpuBackend>>::alloc_paged_pool(&mut ctx, 2, 16, 3, 7);
    let storage = <KvFp16 as KvLayer<CpuBackend>>::physical_storage().unwrap();
    assert_eq!(storage, KvCacheStorage::F32);
    assert_eq!(k.len(), 2 * 16 * 3 * 7);
    assert_eq!(v.len(), k.len());
    assert_eq!(
        storage.bytes_per_token(1, 3, 7).unwrap() * 32,
        (std::mem::size_of_val(k.as_slice()) + std::mem::size_of_val(v.as_slice())) as u64
    );
    assert_ne!(
        CpuBackend::activation_elem_size_bytes(),
        KvFp16::BYTES_PER_ELEM
    );
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires an actual Metal device; allocates only 5376 payload bytes"]
fn physical_kv_storage_metal_default_marker_matches_allocated_f32_payload() {
    use crate::backend::{metal::MetalBackend, Dtype};
    let mut ctx = MetalBackend::new_context(&Default::default()).unwrap();
    let (k, v) = <KvFp16 as KvLayer<MetalBackend>>::alloc_paged_pool(&mut ctx, 2, 16, 3, 7);
    let storage = <KvFp16 as KvLayer<MetalBackend>>::physical_storage().unwrap();
    assert_eq!(storage, KvCacheStorage::F32);
    for buffer in [&k, &v] {
        assert_eq!(buffer.dtype(), Dtype::F32);
        assert_eq!(buffer.len(), 2 * 16 * 3 * 7);
        assert_eq!(buffer.raw().length(), (buffer.len() * 4) as u64);
    }
    assert_eq!(
        storage.bytes_per_token(1, 3, 7).unwrap() * 32,
        k.raw().length() + v.raw().length()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires actual CUDA allocation; no model or numerical kernels"]
fn physical_kv_storage_cuda_values_and_scales_match_actual_allocations() {
    use crate::backend::{cuda::CudaBackend, Dtype};
    use cudarc::driver::DeviceSlice;
    let mut ctx = CudaBackend::new_context(&Default::default()).unwrap();
    let (k, v) = <KvFp16 as KvLayer<CudaBackend>>::alloc_paged_pool(&mut ctx, 2, 16, 3, 7);
    assert_eq!(k.dtype(), Dtype::F16);
    assert_eq!(v.dtype(), Dtype::F16);
    assert_eq!(
        <KvFp16 as KvLayer<CudaBackend>>::physical_storage(),
        Some(KvCacheStorage::F16)
    );
    assert_eq!(
        KvCacheStorage::F16.bytes_per_token(1, 3, 7).unwrap() * 32,
        ((k.len() + v.len()) * 2) as u64
    );
    let pool = <KvInt8 as KvLayer<CudaBackend>>::alloc_paged_pool(&mut ctx, 2, 16, 3, 7);
    let actual_bytes = pool.k.buffer().len()
        + pool.v.buffer().len()
        + 2 * (pool.k_scales.buffer().len() + pool.v_scales.buffer().len());
    assert_eq!(
        <KvInt8 as KvLayer<CudaBackend>>::physical_storage()
            .unwrap()
            .bytes_per_token(1, 3, 7)
            .unwrap()
            * 32,
        actual_bytes as u64
    );
    CudaBackend::try_sync(&mut ctx).unwrap();
}
