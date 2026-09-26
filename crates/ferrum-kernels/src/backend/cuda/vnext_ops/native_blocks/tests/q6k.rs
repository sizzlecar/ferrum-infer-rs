//! Strict Q6 ABI qualification. Timing is synthetic and never a model gate.
use super::*;
use cudarc::driver::{sys, CudaGraph, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{ElementType, WeightId};

mod fixture;
mod timing;

fn generic_control(kernels: &CudaNativeBlockKernels) -> CudaNativeBlockKernels {
    let mut old = kernels.clone();
    old.linear_q6k_f16 = kernels.linear_f16.clone();
    old.linear_q6k_tiled_f16 = kernels.linear_tiled_f16.clone();
    old.linear_q6k_f32 = kernels.linear_f32.clone();
    old.linear_q6k_tiled_f32 = kernels.linear_tiled_f32.clone();
    old.linear_q6k_f32_f16 = kernels.linear_f32_f16.clone();
    old.linear_q6k_tiled_f32_f16 = kernels.linear_tiled_f32_f16.clone();
    old
}

#[test]
fn q6k_template_reference_is_bounded_and_keeps_f32_inputs() {
    let (raw, decoded) = fixture::templates(768);
    assert_eq!(raw.len(), fixture::TEMPLATES * 3 * 210);
    assert_eq!(decoded.len(), fixture::TEMPLATES * 768);
    assert!(decoded.iter().all(|x| x.is_finite()));
    assert!(decoded.iter().any(|x| *x < 0.0));
    assert!(decoded.iter().any(|x| *x > 0.0));
    let first = fixture::input_values::<f32>(2, 768, 0);
    let second = fixture::input_values::<f32>(2, 768, 1);
    assert_ne!(first, second);
    assert!(first[3..3 + 2 * 768]
        .iter()
        .any(|x| f16::from_f32(*x).to_f32() != *x));
}

#[test]
#[ignore = "requires actual CUDA; selected exports, F64 and partition guards"]
fn q6k_production_dtype_and_midrow_boundaries_on_cuda() {
    use ElementType::{F16, F32};
    let context = CudaContext::new(0).unwrap();
    for rows in [1, 4, 8, 9, 31, 32] {
        let tiled = if rows == 1 { "" } else { "tiled_" };
        super::shared_dispatch::check::<f16, f16>(
            &context,
            GgufBlockFormat::Q6K,
            rows,
            768,
            17,
            F16,
            F16,
            &if rows < 32 {
                format!("vnext_gguf_linear_q6k_{tiled}f16")
            } else {
                "vnext_gguf_linear_tiled_f16".to_owned()
            },
        );
        super::shared_dispatch::check::<f32, f32>(
            &context,
            GgufBlockFormat::Q6K,
            rows,
            768,
            17,
            F32,
            F32,
            &format!("vnext_gguf_linear_q6k_{tiled}f32"),
        );
        super::shared_dispatch::check::<f32, f16>(
            &context,
            GgufBlockFormat::Q6K,
            rows,
            768,
            17,
            F32,
            F16,
            &format!("vnext_gguf_linear_q6k_{tiled}f32_f16"),
        );
    }
    // The already-qualified substantial F16 M32 route remains shared GEMM.
    super::shared_dispatch::check::<f16, f16>(
        &context,
        GgufBlockFormat::Q6K,
        32,
        4096,
        4096,
        F16,
        F16,
        "vnext_gguf_gemm_q6k_f16",
    );
}

#[test]
#[ignore = "requires actual CUDA; all strict dtypes and changed-input graph replay"]
fn q6k_fixed_abi_matches_generic_bits_two_generations_on_cuda() {
    let context = CudaContext::new(0).unwrap();
    let kernels = CudaNativeBlockKernels::load(&context).unwrap();
    let old = generic_control(&kernels);
    let stream = context.new_stream().unwrap();
    for rows in [1, 4, 8, 9, 31, 32] {
        let mut f16_case = fixture::Fixture::<f16, f16>::new(
            &stream,
            rows,
            768,
            17,
            ElementType::F16,
            ElementType::F16,
        );
        fixture::qualify(&mut f16_case, &stream, &old, &kernels);
        let mut f32_case = fixture::Fixture::<f32, f32>::new(
            &stream,
            rows,
            768,
            17,
            ElementType::F32,
            ElementType::F32,
        );
        fixture::qualify(&mut f32_case, &stream, &old, &kernels);
        let mut mixed_case = fixture::Fixture::<f32, f16>::new(
            &stream,
            rows,
            768,
            17,
            ElementType::F32,
            ElementType::F16,
        );
        fixture::qualify(&mut mixed_case, &stream, &old, &kernels);
    }
}
