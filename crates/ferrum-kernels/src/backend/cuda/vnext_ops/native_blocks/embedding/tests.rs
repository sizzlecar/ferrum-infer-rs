use super::*;
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use ferrum_interfaces::vnext::{HadamardTransformSpec, WeightId};
use std::num::NonZeroU32;
fn part() -> weights::MatrixPart {
    weights::MatrixPart {
        component_id: WeightId::new("weight.embedding").unwrap(),
        format: weights::MatrixFormat::Block(GgufBlockFormat::Q4K),
        rows: 32,
        columns: 512,
        output_offset: 0,
        transform: None,
        signs_region: None,
    }
}
#[test]
fn cuda_selected_embedding_native_preserves_full_abi_dtype_and_inverse_transform() {
    for output in [ElementType::F16, ElementType::F32] {
        let mut part = part();
        let plain = lookup_plan(&part, 3, output).unwrap();
        assert_eq!(
            plain.entry,
            if output == ElementType::F16 {
                F16_ENTRY
            } else {
                F32_ENTRY
            }
        );
        assert_eq!(&plain.parameters[..3], &[3, 512, 32]);
        let work = selected(
            &part,
            [2, 3],
            5,
            output,
            0,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        work.validate_command(5, 2, 0).unwrap();
        part.transform = Some(HadamardTransformSpec {
            block_size: NonZeroU32::new(256).unwrap(),
            signs: HadamardSigns::Identity,
            application: HadamardApplication::AfterEmbeddingLookup,
        });
        let work = selected(
            &part,
            [2, 3],
            5,
            output,
            5 * 512 * 4,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        work.validate_command(5, 4, 0).unwrap();
        work.algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(&work)
            .unwrap();
        let template = SelectedReplayAlgorithmTemplateV1::from_selected(&work, 5, 4, 0).unwrap();
        part.transform.as_mut().unwrap().block_size = NonZeroU32::new(128).unwrap();
        let changed = selected(
            &part,
            [2, 3],
            5,
            output,
            5 * 512 * 4,
            SloStructuredCostCapture::HostSettledV1,
        )
        .unwrap();
        assert!(template.validate_binding(&changed).is_err());
        assert!(selected(
            &part,
            [2, 3],
            5,
            output,
            1,
            SloStructuredCostCapture::HostSettledV1
        )
        .is_none());
    }
}
#[test]
fn cuda_selected_embedding_native_rejects_partial_tables_sign_mismatch_and_is_lazy_off() {
    let mut part = part();
    let lazy =
        std::iter::from_fn(|| -> Option<u64> { panic!("Off enumerated native embedding rows") });
    assert!(selected(
        &part,
        lazy,
        1,
        ElementType::F16,
        0,
        SloStructuredCostCapture::Disabled
    )
    .is_none());
    part.output_offset = 1;
    assert!(selected(
        &part,
        [1],
        1,
        ElementType::F16,
        0,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    part.output_offset = 0;
    part.signs_region = Some(1);
    assert!(selected(
        &part,
        [1],
        1,
        ElementType::F16,
        0,
        SloStructuredCostCapture::HostSettledV1
    )
    .is_none());
    part.signs_region = None;
    let counts = [65535, 1];
    let work = selected(
        &part,
        counts,
        65536,
        ElementType::F32,
        0,
        SloStructuredCostCapture::HostSettledV1,
    )
    .unwrap();
    work.validate_command(65536, 2, 0).unwrap();
}
