use super::*;
use ferrum_interfaces::vnext::WeightId;
fn part(offset: u32) -> MatrixPart {
    MatrixPart {
        component_id: WeightId::new(format!("q8.pair.{offset}")).unwrap(),
        format: MatrixFormat::Block(crate::gguf_blocks::GgufBlockFormat::Q8_0),
        rows: 32,
        columns: 4096,
        output_offset: offset,
        transform: None,
        signs_region: None,
    }
}
#[test]
fn q8_pair_plan_preserves_two_physical_ranges_and_qualified_geometry() {
    let parts = [part(12288), part(12320)];
    for rows in [4, 8] {
        let plan = PairPlan::select(
            &parts[0],
            &parts[1],
            rows,
            4096,
            12352,
            ElementType::F16,
            ElementType::F16,
        )
        .unwrap();
        assert_eq!(
            plan.parameters(),
            [rows, 4096, 32, 12352, 12288, 12320, 8, 32, 34]
        );
        assert_eq!(dispatches(&parts, rows, 4096, 12352), 1);
        let reversed = PairPlan::select(
            &parts[1],
            &parts[0],
            rows,
            4096,
            12352,
            ElementType::F16,
            ElementType::F16,
        )
        .unwrap();
        assert_eq!(reversed.offsets, [12320, 12288]);
    }
    for rows in [1, 3, 5, 7, 9, 16, u16::MAX.into()] {
        assert!(PairPlan::select(
            &parts[0],
            &parts[1],
            rows,
            4096,
            12352,
            ElementType::F16,
            ElementType::F16
        )
        .is_none());
        assert_eq!(dispatches(&parts, rows, 4096, 12352), 2);
    }
}
#[test]
fn q8_pair_plan_rejects_alias_dtype_transform_and_extent_changes() {
    let parts = [part(12288), part(12320)];
    let mut transformed = parts.clone();
    transformed[1].transform = Some(ferrum_interfaces::vnext::HadamardTransformSpec {
        block_size: std::num::NonZeroU32::new(256).unwrap(),
        signs: ferrum_interfaces::vnext::HadamardSigns::Identity,
        application: ferrum_interfaces::vnext::HadamardApplication::BeforeMatmul {
            input_permutation: None,
        },
    });
    assert!(PairPlan::select(
        &transformed[0],
        &transformed[1],
        8,
        4096,
        12352,
        ElementType::F16,
        ElementType::F16
    )
    .is_none());
    assert_eq!(
        dispatches(&transformed, 8, 4096, 12352),
        3,
        "retain transform dispatch"
    );
    for (input, output) in [
        (ElementType::F32, ElementType::F16),
        (ElementType::F16, ElementType::F32),
    ] {
        assert!(PairPlan::select(&parts[0], &parts[1], 8, 4096, 12352, input, output).is_none());
    }
    for variant in 0..7 {
        let mut second = parts[1].clone();
        match variant {
            0 => second.output_offset = 12319,
            1 => second.output_offset = u32::MAX,
            2 => second.rows = 31,
            3 => second.columns = 2048,
            4 => second.format = MatrixFormat::DenseF16,
            5 => second.signs_region = Some(2),
            _ => second.output_offset = 12321,
        }
        assert!(PairPlan::select(
            &parts[0],
            &second,
            8,
            4096,
            12352,
            ElementType::F16,
            ElementType::F16
        )
        .is_none());
    }
    assert!(PairPlan::select(
        &parts[0],
        &parts[1],
        8,
        2048,
        12352,
        ElementType::F16,
        ElementType::F16
    )
    .is_none());
}
