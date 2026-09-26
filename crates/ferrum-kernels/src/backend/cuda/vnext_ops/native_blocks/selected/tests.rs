use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::vnext::{HadamardTransformSpec, WeightId};
use std::num::NonZeroU32;

fn part(format: weights::MatrixFormat, outputs: u32, columns: u32) -> weights::MatrixPart {
    weights::MatrixPart {
        component_id: WeightId::new("component.native-selected-test").unwrap(),
        format,
        rows: outputs,
        columns,
        output_offset: 0,
        transform: None,
        signs_region: None,
    }
}

#[test]
fn cuda_selected_native_matrix_preserves_real_selector_boundaries_and_dtype() {
    use weights::MatrixFormat::Block;
    let cases = [
        (
            GgufBlockFormat::Q4K,
            1,
            512,
            256,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_linear_q4k_f16",
            [64, 1, 1],
        ),
        (
            GgufBlockFormat::Q4K,
            8,
            512,
            256,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_linear_q4k_tiled_f16",
            [64, 1, 1],
        ),
        (
            GgufBlockFormat::Q5K,
            31,
            4096,
            2560,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_linear_q5k_tiled_f16",
            [640, 4, 1],
        ),
        (
            GgufBlockFormat::Q5K,
            32,
            4096,
            2560,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_gemm_q5k_f16",
            [40, 1, 1],
        ),
        (
            GgufBlockFormat::Q6K,
            32,
            256,
            512,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_linear_tiled_f16",
            [128, 4, 1],
        ),
        (
            GgufBlockFormat::Q6K,
            32,
            4096,
            4096,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_gemm_q6k_f16",
            [64, 1, 1],
        ),
        (
            GgufBlockFormat::Q6K,
            8,
            256,
            512,
            ElementType::F32,
            ElementType::F32,
            "vnext_gguf_linear_q6k_tiled_f32",
            [128, 1, 1],
        ),
        (
            GgufBlockFormat::Q6K,
            8,
            256,
            512,
            ElementType::F32,
            ElementType::F16,
            "vnext_gguf_linear_q6k_tiled_f32_f16",
            [128, 1, 1],
        ),
        (
            GgufBlockFormat::Q8_0,
            128,
            256,
            512,
            ElementType::F16,
            ElementType::F16,
            "vnext_gguf_linear_tiled_f16",
            [128, 16, 1],
        ),
    ];
    for (format, rows, k, n, input, output, entry, grid) in cases {
        let p = part(Block(format), n, k);
        let plan = linear_launch::select(&p, rows, n, input, output).unwrap();
        assert_eq!(plan.kernel.entry(), entry);
        assert_eq!(
            [
                plan.config.grid_dim.0,
                plan.config.grid_dim.1,
                plan.config.grid_dim.2
            ],
            grid
        );
    }
}

#[test]
fn cuda_selected_native_matrix_work_is_numeric_and_physical_leaf_count_is_preserved() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let p = part(weights::MatrixFormat::Block(GgufBlockFormat::Q6K), 7, 256);
    let packed = linear(&[p.clone()], [8], 8, 7, ElementType::F32, 0, on).unwrap();
    let leaves = linear(&[p.clone()], [1; 8], 8, 7, ElementType::F32, 0, on).unwrap();
    packed.validate_command(8, 1, 0).unwrap();
    leaves.validate_command(8, 8, 0).unwrap();
    assert!(packed.validate_command(8, 8, 0).is_err());
    assert_eq!(packed.work().logical_units, 56);
    assert_eq!(packed.work().padded_units, 64);
    assert_eq!(packed.work().inner_work_units, 56 * 256);
    assert_eq!(
        packed.work().inner_work_units,
        leaves.work().inner_work_units
    );
    assert_eq!(packed.work().grid_blocks, 2);
    assert_eq!(leaves.work().grid_blocks, 16);
    assert_ne!(packed.family_signature(), leaves.family_signature());
    let a = linear(&[p], [2], 2, 7, ElementType::F32, 0, on).unwrap();
    let b = linear(
        &[part(
            weights::MatrixFormat::Block(GgufBlockFormat::Q6K),
            11,
            512,
        )],
        [7],
        7,
        11,
        ElementType::F32,
        0,
        on,
    )
    .unwrap();
    assert_eq!(a.family_signature(), b.family_signature());
    assert_ne!(a.work(), b.work());
    let resident =
        ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1::from_selected(
            &a, 2, 1, 0,
        )
        .unwrap();
    resident.validate_binding(&a).unwrap();
    let changed_k = linear(
        &[part(
            weights::MatrixFormat::Block(GgufBlockFormat::Q6K),
            7,
            512,
        )],
        [2],
        2,
        7,
        ElementType::F32,
        0,
        on,
    )
    .unwrap();
    assert_eq!(a.family_signature(), changed_k.family_signature());
    assert_eq!(a.work().grid_blocks, changed_k.work().grid_blocks);
    assert!(
        resident.validate_binding(&changed_k).is_err(),
        "resident K is a fixed launch parameter"
    );
    for evidence in [packed, leaves, a, b] {
        evidence
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(&evidence)
            .unwrap();
    }
}

#[test]
fn cuda_selected_native_matrix_transform_has_separate_launch_and_exact_scratch() {
    let on = SloStructuredCostCapture::HostSettledV1;
    let mut p = part(weights::MatrixFormat::Block(GgufBlockFormat::Q6K), 7, 256);
    p.transform = Some(HadamardTransformSpec {
        block_size: NonZeroU32::new(128).unwrap(),
        signs: HadamardSigns::Identity,
        application: HadamardApplication::BeforeMatmul {
            input_permutation: None,
        },
    });
    let a = linear(&[p.clone()], [3], 3, 7, ElementType::F16, 3 * 256 * 4, on).unwrap();
    assert!(linear(
        &[p.clone()],
        [3],
        3,
        7,
        ElementType::F16,
        3 * 256 * 4 - 1,
        on,
    )
    .is_none());
    a.validate_command(3, 2, 0).unwrap();
    assert_eq!(a.work().peak_scratch_bytes, 3 * 256 * 4);
    assert_eq!(a.work().staged_weight_bytes, 0);
    p.transform.as_mut().unwrap().block_size = NonZeroU32::new(256).unwrap();
    let b = linear(&[p.clone()], [3], 3, 7, ElementType::F16, 3 * 256 * 4, on).unwrap();
    assert_ne!(a.family_signature(), b.family_signature());
    p.transform.as_mut().unwrap().application = HadamardApplication::AfterEmbeddingLookup;
    assert!(linear(&[p.clone()], [3], 3, 7, ElementType::F16, 3 * 256 * 4, on).is_none());
    p.transform = None;
    p.signs_region = Some(0);
    assert!(linear(&[p], [3], 3, 7, ElementType::F16, 3 * 256 * 4, on).is_none());
}

#[test]
fn cuda_selected_native_matrix_disabled_is_lazy_and_invalid_work_stays_absent() {
    let p = part(weights::MatrixFormat::Block(GgufBlockFormat::Q6K), 7, 256);
    let lazy = std::iter::once_with(|| panic!("Disabled must not enumerate launch metadata"));
    assert!(linear(
        &[p.clone()],
        lazy,
        1,
        7,
        ElementType::F32,
        0,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    let on = SloStructuredCostCapture::HostSettledV1;
    for rows in [0, 65536, u32::MAX] {
        assert!(linear(
            &[p.clone()],
            [rows],
            u64::from(rows),
            7,
            ElementType::F32,
            0,
            on
        )
        .is_none());
    }
    assert!(linear(&[p.clone()], [1], 1, 6, ElementType::F32, 0, on).is_none());
    assert!(linear(&[p.clone()], [1], 1, 7, ElementType::U32, 0, on).is_none());
    let overflow = part(weights::MatrixFormat::DenseF16, u32::MAX, u32::MAX);
    assert!(linear(
        &[overflow],
        [65535],
        65535,
        u32::MAX,
        ElementType::F32,
        0,
        on
    )
    .is_none());
}
