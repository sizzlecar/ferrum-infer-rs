use super::*;
pub(super) fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
pub(in crate::qwen3_vl_vision) use super::super::test_fixture::{
    config, limits, qwen35_config, Original,
};
pub(in crate::qwen3_vl_vision) fn host(limit: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            limit,
            HostMemoryClassLimits {
                input_preprocess_bytes: limit,
                output_bytes: limit,
                transfer_staging_bytes: limit,
                weight_backing_bytes: limit,
            },
        )
        .unwrap(),
    )
}
pub(super) fn patches(c: &Qwen3VLVisionComfyConfig, which: usize) -> Vec<f32> {
    let n = (c.rows().unwrap() * c.architecture().patch_columns().unwrap()) as usize;
    (0..n)
        .map(|i| (((i * 13 + which * 19) % 41) as f32 - 20.) / 16. + which as f32 / 128.)
        .collect()
}
pub(super) fn input(
    host: &HostMemoryAccount,
    c: &Qwen3VLVisionComfyConfig,
    values: &[f32],
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 4)
        .unwrap();
    for (b, v) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
        b.copy_from_slice(&v.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(Qwen3VLVisionComfyFamily::PATCHES_ID),
        ResolvedTensorSpec::new(
            vec![c.rows().unwrap(), c.architecture().patch_columns().unwrap()],
            ElementType::F32,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap(),
        bytes.freeze(),
    )
    .unwrap()
}
pub(super) fn floats(t: &ManagedHostTensor) -> Vec<f32> {
    t.bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
