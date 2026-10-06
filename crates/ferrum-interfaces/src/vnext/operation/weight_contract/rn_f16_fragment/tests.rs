use super::*;
use crate::vnext::{
    ModelFamilyId, PhysicalStorageLayout, PhysicalWeightComponentBinding, PhysicalWeightLayout,
    WeightComponentRole, WeightComponentSpec, WeightSchema, WeightTensorSpec,
};

fn fixture(format: RnF16FragmentSourceFormatV1, dimensions: &[u64]) -> WeightSchema {
    let plan = RnF16FragmentPlanV1::from_dimensions(format, dimensions).unwrap();
    let dense = WeightId::new("fixture.dense").unwrap();
    let packet = WeightId::new("fixture.packet").unwrap();
    WeightSchema {
        format_id: WeightFormatId::new("fixture.rn-fragment.format").unwrap(),
        layout_id: WeightLayoutId::new("fixture.rn-fragment.layout").unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![
            WeightComponentSpec {
                id: dense.clone(),
                role: WeightComponentRole::Values,
                external_names: vec!["derived.dense".into()],
                dimensions: dimensions.to_vec(),
                encoding: WeightEncoding::Dense {
                    element_type: ElementType::F16,
                },
                required: true,
            },
            WeightComponentSpec {
                id: packet.clone(),
                role: WeightComponentRole::PackedValues,
                external_names: vec!["derived.fragment".into()],
                dimensions: plan.packed_dimensions().to_vec(),
                encoding: plan.packed_encoding(),
                required: true,
            },
        ],
        tensors: vec![WeightTensorSpec {
            id: WeightId::new("fixture.projection").unwrap(),
            dimensions: dimensions.to_vec(),
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
                dense_values: PhysicalWeightComponentBinding::exact_contiguous(dense),
                fragment_values: PhysicalWeightComponentBinding::exact_contiguous(packet),
                source_format: format,
            },
            required: true,
        }],
    }
}

#[test]
fn rn_fragment_plan_exact_spans_and_one_whole_n_tail() {
    for (format, source_row, tile_bytes) in [
        (RnF16FragmentSourceFormatV1::Q4K, 144, 384),
        (RnF16FragmentSourceFormatV1::Q5K, 176, 448),
        (RnF16FragmentSourceFormatV1::Q6K, 210, 512),
    ] {
        let p = RnF16FragmentPlanV1::new(format, 18, 256).unwrap();
        assert_eq!(p.packing_abi(), 0x524e4631);
        assert_eq!((p.n(), p.k(), p.source_format()), (18, 256, format));
        assert_eq!(p.source_row_bytes(), source_row);
        assert_eq!(p.source_bytes(), 18 * source_row);
        assert_eq!(p.dense_bytes(), 18 * 256 * 2);
        assert_eq!(p.packed_dimensions(), [2, 8]);
        assert_eq!(p.packed_bytes(), 2 * 8 * tile_bytes);
        assert_eq!(
            RnF16FragmentPlanV1::from_dimensions(format, &[2, 9, 256]).unwrap(),
            p
        );
        // A two-part gate/up is tiled once, even when each part is not N16 aligned.
        let whole = RnF16FragmentPlanV1::new(format, 6, 256).unwrap();
        let part = RnF16FragmentPlanV1::new(format, 3, 256).unwrap();
        assert_eq!(whole.packed_bytes(), part.packed_bytes());
        assert_eq!(whole.source_bytes(), 2 * part.source_bytes());
    }
}

#[test]
fn rn_fragment_plan_rejects_invalid_shape_and_overflow_before_allocation() {
    let f = RnF16FragmentSourceFormatV1::Q6K;
    for (n, k) in [
        (0, 256),
        (1, 0),
        (1, 255),
        (1, 257),
        (u64::MAX, 256),
        (u64::from(u32::MAX) + 1, 256),
        (1, u64::from(u32::MAX) + 1),
        (u64::from(u32::MAX), u64::from(u32::MAX) - 255),
    ] {
        assert!(RnF16FragmentPlanV1::new(f, n, k).is_err(), "{n}/{k}");
    }
    for dims in [
        vec![],
        vec![256],
        vec![1, 1, 1, 256],
        vec![2, 0, 256],
        vec![u64::MAX, 2, 256],
    ] {
        assert!(RnF16FragmentPlanV1::from_dimensions(f, &dims).is_err());
    }
    assert!(RnF16FragmentPlanV1::new(f, 1, 256).is_ok());
}

#[test]
fn rn_fragment_schema_declares_both_resources_and_distinct_wire_identity() {
    let family = ModelFamilyId::new("fixture.family").unwrap();
    let mut fingerprints = std::collections::BTreeSet::new();
    for f in [
        RnF16FragmentSourceFormatV1::Q4K,
        RnF16FragmentSourceFormatV1::Q5K,
        RnF16FragmentSourceFormatV1::Q6K,
    ] {
        let schema = fixture(f, &[2, 9, 256]);
        schema.validate(&family).unwrap();
        let refs = schema
            .physical_component_refs(&schema.tensors[0].id)
            .unwrap();
        assert_eq!(refs.len(), 2);
        let p = RnF16FragmentPlanV1::new(f, 18, 256).unwrap();
        let physical = refs
            .iter()
            .map(|c| c.physical_bytes().unwrap())
            .sum::<u64>();
        assert_eq!(physical, p.dense_bytes() + p.packed_bytes());
        assert!(physical > schema.tensors[0].logical_bytes().unwrap());
        let decoded: WeightSchema =
            serde_json::from_slice(&serde_json::to_vec(&schema).unwrap()).unwrap();
        assert_eq!(decoded, schema);
        fingerprints.insert(schema.fingerprint().unwrap());
    }
    assert_eq!(fingerprints.len(), 3);
}

#[test]
fn rn_fragment_schema_rejects_swapped_aliased_wrong_format_or_extra_storage() {
    let family = ModelFamilyId::new("fixture.family").unwrap();
    let original = fixture(RnF16FragmentSourceFormatV1::Q4K, &[17, 256]);
    for case in 0..9 {
        let mut bad = original.clone();
        match case {
            0 => bad.tensors[0].logical_element_type = ElementType::F32,
            1 => {
                if let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
                    dense_values,
                    fragment_values,
                    ..
                } = &mut bad.tensors[0].physical_layout
                {
                    std::mem::swap(dense_values, fragment_values);
                }
            }
            2 => {
                if let PhysicalWeightLayout::RnF16DenseAndFragmentV1 {
                    dense_values,
                    fragment_values,
                    ..
                } = &mut bad.tensors[0].physical_layout
                {
                    fragment_values.component_id = dense_values.component_id.clone();
                }
            }
            3 => {
                if let PhysicalWeightLayout::RnF16DenseAndFragmentV1 { source_format, .. } =
                    &mut bad.tensors[0].physical_layout
                {
                    *source_format = RnF16FragmentSourceFormatV1::Q6K;
                }
            }
            4 => bad.components[1].dimensions[0] += 1,
            5 => bad.components[0].dimensions[0] += 1,
            6 => bad.components[1].role = WeightComponentRole::Values,
            7 => {
                bad.components[1].encoding = WeightEncoding::BlockQuantized(
                    RnF16FragmentPlanV1::new(RnF16FragmentSourceFormatV1::Q4K, 17, 256)
                        .unwrap()
                        .source_block_spec(),
                )
            }
            8 => {
                if let PhysicalWeightLayout::RnF16DenseAndFragmentV1 { dense_values, .. } =
                    &mut bad.tensors[0].physical_layout
                {
                    dense_values.storage = PhysicalStorageLayout::Strided {
                        strides_in_elements: vec![256, 1],
                        padding: super::super::PhysicalWeightPadding::Exact,
                    };
                }
            }
            _ => unreachable!(),
        }
        assert!(bad.validate(&family).is_err(), "case {case}");
    }
}
