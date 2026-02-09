use ferrum_types::*;

#[test]
fn dtype_helpers_and_display() {
    assert_eq!(DataType::FP32.size_bytes(), 4);
    assert!(DataType::FP16.is_float());
    assert!(DataType::INT8.is_integer());
    assert!(DataType::FP8.is_quantized());
    assert_eq!(DataType::UINT8.to_string(), "uint8");
}

#[test]
fn device_helpers() {
    assert!(!Device::CPU.is_gpu());
    assert_eq!(Device::CUDA(0).index(), Some(0));
    assert_eq!(Device::ROCm(1).index(), Some(1));
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        assert!(Device::Metal.is_gpu());
        assert_eq!(Device::Metal.to_string(), "metal");
    }
}
