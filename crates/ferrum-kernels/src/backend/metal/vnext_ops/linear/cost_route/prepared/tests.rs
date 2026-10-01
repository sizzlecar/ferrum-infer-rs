use super::*;
use ferrum_interfaces::vnext::*;
use std::collections::BTreeMap;

// Typed, metadata-only native block fixtures; no weights/device allocation.
fn weight(
    gate: bool,
    hidden: u64,
    intermediate: u64,
    formats: &[GgufBlockFormat],
) -> ResolvedWeightBinding {
    let name = if gate { "gate_up" } else { "down" };
    let weight_id = WeightId::new(name).unwrap();
    let rows = if gate { intermediate } else { hidden };
    let columns = if gate { hidden } else { intermediate };
    let mut components = Vec::new();
    let mut leaves = Vec::new();
    for (index, &format) in formats.iter().enumerate() {
        let id = WeightId::new(format!("{name}.{index}")).unwrap();
        let spec = BlockQuantizationSpec {
            format_id: QuantizationFormatId::new(format.format_id()).unwrap(),
            logical_values_per_block: format.block_values() as u32,
            bytes_per_block: format.block_bytes() as u32,
        };
        let mut dimensions = vec![rows, columns / u64::from(spec.logical_values_per_block)];
        if gate {
            dimensions.insert(0, 1);
        }
        components.push(WeightComponentSpec {
            id: id.clone(),
            role: WeightComponentRole::PackedValues,
            external_names: vec![format!("{name}.{index}")],
            dimensions,
            encoding: WeightEncoding::BlockQuantized(spec),
            required: true,
        });
        leaves.push(PhysicalWeightLayout::BlockQuantized {
            blocks: PhysicalWeightComponentBinding {
                component_id: id,
                storage: PhysicalStorageLayout::exact_contiguous(),
            },
            block_axis: if gate { 2 } else { 1 },
            block_padding: PhysicalWeightPadding::Exact,
        });
    }
    let dimensions = if gate {
        vec![2, intermediate, hidden]
    } else {
        vec![hidden, intermediate]
    };
    let physical_layout = if gate {
        PhysicalWeightLayout::Composite {
            parts: leaves
                .into_iter()
                .enumerate()
                .map(|(index, layout)| CompositeWeightPart {
                    layout: Box::new(layout),
                    logical_offsets: vec![index as u64, 0, 0],
                    extents: vec![1, intermediate, hidden],
                })
                .collect(),
        }
    } else {
        leaves.pop().unwrap()
    };
    let schema = WeightSchema {
        format_id: "weight-format.gguf.native-block"
            .to_owned()
            .try_into()
            .unwrap(),
        layout_id: "layout.gguf.native-block".to_owned().try_into().unwrap(),
        version: ContractVersion::new(1, 0),
        components,
        tensors: vec![WeightTensorSpec {
            id: weight_id.clone(),
            dimensions,
            logical_element_type: ElementType::F16,
            physical_layout,
            required: true,
        }],
    };
    ResolvedWeightBinding::from_schema(&schema, &weight_id).unwrap()
}

fn bindings(
    hidden: u64,
    intermediate: u64,
    formats: [GgufBlockFormat; 3],
) -> Vec<ResolvedValueBinding> {
    let gate = weight(true, hidden, intermediate, &formats[..2]);
    let down = weight(false, hidden, intermediate, &formats[2..]);
    [
        (ResolvedValueRole::Input, 0, vec![768, hidden], None),
        (
            ResolvedValueRole::Input,
            1,
            vec![2, intermediate, hidden],
            Some(gate),
        ),
        (
            ResolvedValueRole::Input,
            2,
            vec![hidden, intermediate],
            Some(down),
        ),
        (ResolvedValueRole::Output, 0, vec![768, hidden], None),
    ]
    .into_iter()
    .map(|(role, ordinal, dimensions, weight)| {
        let name = format!("{role:?}.{ordinal}");
        let tensor = ResolvedTensorSpec::new(
            dimensions,
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let storage = match &weight {
            Some(weight) => ResolvedValueStorage::composite(
                weight
                    .components()
                    .iter()
                    .map(|component| {
                        ResolvedStorageComponent::new(
                            Some(component.component_id().clone()),
                            ResourceId::new(format!("resource.{}", component.component_id()))
                                .unwrap(),
                            0,
                            component.physical_bytes().unwrap(),
                            component.physical_element_type(),
                        )
                        .unwrap()
                    })
                    .collect(),
            )
            .unwrap(),
            None => ResolvedValueStorage::single(
                ResourceId::new(name.clone()).unwrap(),
                0,
                tensor.minimum_storage_bytes().unwrap(),
                ElementType::F16,
            )
            .unwrap(),
        };
        ResolvedValueBinding::new(
            ProgramValueId::new(name).unwrap(),
            role,
            ordinal,
            tensor,
            if role == ResolvedValueRole::Output {
                TensorAccess::Write
            } else {
                TensorAccess::Read
            },
            AliasPolicy::NoAlias,
            if weight.is_some() {
                BufferUsage::Weights
            } else {
                BufferUsage::Activations
            },
            weight,
            storage,
        )
        .unwrap()
    })
    .collect()
}

fn attributes(hidden: u64, intermediate: u64) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (
            AttributeId::new("hidden_size").unwrap(),
            SemanticValue::Unsigned(hidden),
        ),
        (
            AttributeId::new("intermediate_size").unwrap(),
            SemanticValue::Unsigned(intermediate),
        ),
    ])
}

fn dense_fixture(
    input: u64,
    output: u64,
    format: GgufBlockFormat,
) -> (
    BTreeMap<AttributeId, SemanticValue>,
    Vec<ResolvedValueBinding>,
) {
    let weight = weight(false, output, input, &[format]);
    let values = [
        (ResolvedValueRole::Input, 0, vec![768, input], None),
        (
            ResolvedValueRole::Input,
            1,
            vec![output, input],
            Some(weight),
        ),
        (ResolvedValueRole::Output, 0, vec![768, output], None),
    ]
    .into_iter()
    .map(|(role, ordinal, dimensions, weight)| {
        let name = format!("dense.{role:?}.{ordinal}");
        let tensor = ResolvedTensorSpec::new(
            dimensions,
            ElementType::F16,
            ResolvedTensorLayout::Contiguous,
        )
        .unwrap();
        let storage = match &weight {
            Some(weight) => {
                let component = &weight.components()[0];
                ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                    Some(component.component_id().clone()),
                    ResourceId::new(name.clone()).unwrap(),
                    0,
                    component.physical_bytes().unwrap(),
                    component.physical_element_type(),
                )
                .unwrap()])
                .unwrap()
            }
            None => ResolvedValueStorage::single(
                ResourceId::new(name.clone()).unwrap(),
                0,
                tensor.minimum_storage_bytes().unwrap(),
                ElementType::F16,
            )
            .unwrap(),
        };
        ResolvedValueBinding::new(
            ProgramValueId::new(name).unwrap(),
            role,
            ordinal,
            tensor,
            if role == ResolvedValueRole::Output {
                TensorAccess::Write
            } else {
                TensorAccess::Read
            },
            AliasPolicy::NoAlias,
            if weight.is_some() {
                BufferUsage::Weights
            } else {
                BufferUsage::Activations
            },
            weight,
            storage,
        )
        .unwrap()
    })
    .collect();
    (
        BTreeMap::from([
            (
                AttributeId::new("in_features").unwrap(),
                SemanticValue::Unsigned(input),
            ),
            (
                AttributeId::new("out_features").unwrap(),
                SemanticValue::Unsigned(output),
            ),
        ]),
        values,
    )
}

#[test]
fn metal_dense_prepared_metadata_retains_exact_abi_and_current_launch_geometry() {
    for format in [
        GgufBlockFormat::Q4K,
        GgufBlockFormat::Q6K,
        GgufBlockFormat::Q8_0,
    ] {
        let (attrs, values) = dense_fixture(512, 2048, format);
        let prepared = PreparedDenseCostData::unprepared(&attrs, &values)
            .unwrap()
            .unwrap();
        for counts in [vec![1], vec![1; 8], vec![3, 5], vec![33], vec![256]] {
            let tokens: u64 = counts.iter().sum();
            let rows: Vec<_> = counts
                .into_iter()
                .map(|count| OperationCostWorkRow {
                    offset: 0,
                    count: std::num::NonZeroU64::new(count).unwrap(),
                    full_input_tokens: std::num::NonZeroU64::new(count).unwrap(),
                })
                .collect();
            for packed in [false, true] {
                let fresh = PreparedDenseCostData::unprepared(&attrs, &values)
                    .unwrap()
                    .unwrap();
                let current = dense_launches(&prepared, &rows, tokens, packed).unwrap();
                let reference = dense_launches(&fresh, &rows, tokens, packed).unwrap();
                assert_eq!(
                    dense_command(rows.len() as u32, tokens, &current).unwrap(),
                    dense_command(rows.len() as u32, tokens, &reference).unwrap()
                );
                assert_eq!(current.len(), if packed { 1 } else { rows.len() });
                for (index, launch) in current.iter().enumerate() {
                    assert_eq!(
                        u64::from(launch.params.rows),
                        if packed {
                            tokens
                        } else {
                            rows[index].count.get()
                        }
                    );
                    assert_eq!(
                        launch.activation_bytes().unwrap(),
                        reference[index].activation_bytes().unwrap()
                    );
                }
            }
        }
        let rows = [OperationCostWorkRow {
            offset: 0,
            count: std::num::NonZeroU64::new(1).unwrap(),
            full_input_tokens: std::num::NonZeroU64::new(1).unwrap(),
        }];
        assert!(dense_launches(&prepared, &rows, u64::from(u32::MAX) + 1, true).is_err());
        let (wrong_attrs, _) = dense_fixture(1024, 2048, format);
        assert!(PreparedDenseCostData::unprepared(&wrong_attrs, &values).is_err());
        assert!(PreparedDenseCostData::unprepared(&attrs, &values[..2]).is_err());
    }
}

#[test]
fn metal_dense_prepared_rows_do_not_reuse_packed_storage_decision() {
    let (attrs, values) = dense_fixture(512, 2048, GgufBlockFormat::Q4K);
    let prepared = PreparedDenseCostData::unprepared(&attrs, &values)
        .unwrap()
        .unwrap();
    let row = OperationCostWorkRow {
        offset: 0,
        count: std::num::NonZeroU64::new(1).unwrap(),
        full_input_tokens: std::num::NonZeroU64::new(1).unwrap(),
    };
    let rows = [row; 8];
    let packed = dense_command(8, 8, &dense_launches(&prepared, &rows, 8, true).unwrap()).unwrap();
    let separate =
        dense_command(8, 8, &dense_launches(&prepared, &rows, 8, false).unwrap()).unwrap();
    assert_eq!(packed.batching(), DeviceBatchingForm::Packed);
    assert_eq!(separate.batching(), DeviceBatchingForm::ParticipantLoop);
    assert_ne!(
        packed.compute_dispatch_count(),
        separate.compute_dispatch_count()
    );
}

#[test]
fn metal_swiglu_preparation_keeps_original_static_abi_and_invalid_input() {
    let hidden = 512;
    let intermediate = 2048;
    let values = bindings(
        hidden,
        intermediate,
        [
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q8_0,
            GgufBlockFormat::Q6K,
        ],
    );
    dense_swiglu_contract()
        .unwrap()
        .descriptor()
        .validate_resolved_bindings(&values)
        .unwrap();
    let attrs = attributes(hidden, intermediate);
    let mut data = PreparedSwiGluCostData::unprepared(&attrs, &values)
        .unwrap()
        .unwrap();
    assert_eq!(data.gate.len(), 2);
    assert_eq!(data.gate[0].output_offset, 0);
    assert_eq!(data.gate[1].output_offset, intermediate as u32);
    assert_eq!(data.staging_bytes, hidden * intermediate * 2);
    assert!(data.classes.is_none());
    data.compile_classes();
    assert!(data.classes.is_some());
    assert!(
        PreparedSwiGluCostData::unprepared(&attributes(hidden * 2, intermediate), &values).is_err()
    );
    assert!(PreparedSwiGluCostData::unprepared(&attrs, &values[..3]).is_err());
    assert!(PreparedSwiGluCostData::unprepared(&BTreeMap::new(), &values).is_err());
}

#[test]
fn metal_swiglu_prepared_route_matches_fresh_at_current_rows_and_staging_boundaries() {
    let device = Device::system_default().expect("actual Metal SwiGLU selector catalog");
    let pipelines = MetalLinearPipelines::new(&device)
        .unwrap()
        .with_structured_capture(ferrum_types::SloStructuredCostCapture::HostSettledV1);
    for formats in [
        [
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q6K,
        ],
        [
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q8_0,
            GgufBlockFormat::Q6K,
        ],
    ] {
        let attrs = attributes(512, 2048);
        let values = bindings(512, 2048, formats);
        let mut prepared = PreparedSwiGluCostData::unprepared(&attrs, &values)
            .unwrap()
            .unwrap();
        prepared.compile_classes();
        assert!(prepared.classes.is_some());
        let mut previous = None;
        for counts in [
            vec![1],
            vec![1; 4],
            vec![1; 8],
            vec![3, 5],
            vec![33],
            vec![255],
            vec![127, 129],
            vec![768],
        ] {
            let tokens = counts.iter().sum();
            let fresh = PreparedSwiGluCostData::unprepared(&attrs, &values)
                .unwrap()
                .unwrap();
            let old = swiglu_command(&pipelines, &fresh, counts.len() as u32, tokens).unwrap();
            let new = swiglu_command(&pipelines, &prepared, counts.len() as u32, tokens).unwrap();
            assert_eq!(
                new, old,
                "physical command must retain current participants/M"
            );
            let old = old.statistical_evidence().unwrap();
            let new = new.statistical_evidence().unwrap();
            assert_eq!(new, old);
            assert_eq!(new.algorithm_work(), old.algorithm_work());
            assert_eq!(
                new.independent_attention_family_v2(),
                old.independent_attention_family_v2()
            );
            assert!(new.algorithm_work().unwrap().is_ok());
            if tokens == 255 {
                assert_eq!(new.work().staged_weight_bytes, 0);
                previous = Some(*new.family_signature());
            }
            if tokens == 256 {
                assert!(new.work().staged_weight_bytes > 0);
                assert_ne!(Some(*new.family_signature()), previous);
            }
        }
        for tokens in [0, u64::MAX, u64::from(u32::MAX) + 1] {
            let fresh = PreparedSwiGluCostData::unprepared(&attrs, &values)
                .unwrap()
                .unwrap();
            assert!(swiglu_command(&pipelines, &fresh, 1, tokens).is_err());
            assert!(swiglu_command(&pipelines, &prepared, 1, tokens).is_err());
        }
        assert!(swiglu_command(&pipelines, &prepared, 0, 1).is_err());
    }
}

#[test]
fn metal_swiglu_prepared_classes_reject_different_matrix_or_partition_abi() {
    let device = Device::system_default().expect("actual Metal SwiGLU selector catalog");
    let pipelines = MetalLinearPipelines::new(&device)
        .unwrap()
        .with_structured_capture(ferrum_types::SloStructuredCostCapture::HostSettledV1);
    let values = bindings(
        512,
        2048,
        [
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q4K,
            GgufBlockFormat::Q6K,
        ],
    );
    let mut correct = PreparedSwiGluCostData::unprepared(&attributes(512, 2048), &values)
        .unwrap()
        .unwrap();
    correct.compile_classes();
    let classes = correct.classes.take().unwrap();
    for values in [
        bindings(
            512,
            2048,
            [
                GgufBlockFormat::Q5K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q6K,
            ],
        ),
        bindings(
            1024,
            2048,
            [
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q6K,
            ],
        ),
    ] {
        let hidden = values[0].tensor().dimensions()[1];
        let other = PreparedSwiGluCostData::unprepared(&attributes(hidden, 2048), &values)
            .unwrap()
            .unwrap();
        let gate = other
            .gate
            .iter()
            .map(|&part| linear_launch(part, 0, 0, 8, other.hidden, other.packed, 0, 0).unwrap())
            .collect::<Vec<_>>();
        let down =
            linear_launch(other.down, 0, 0, 8, other.intermediate, other.hidden, 0, 0).unwrap();
        let activation = swiglu_launch(0, 8 * 2048 * 4, 8, 2048, other.packed).unwrap();
        assert!(selected::swiglu(&pipelines, &gate, down, activation, None, 8, 0).is_some());
        assert!(selected::swiglu_with_prepared_classes(
            &pipelines,
            &gate,
            down,
            activation,
            None,
            8,
            0,
            Some(&classes)
        )
        .is_none());
    }
    let gate = correct
        .gate
        .iter()
        .rev()
        .map(|&part| linear_launch(part, 0, 0, 8, correct.hidden, correct.packed, 0, 0).unwrap())
        .collect::<Vec<_>>();
    let down = linear_launch(
        correct.down,
        0,
        0,
        8,
        correct.intermediate,
        correct.hidden,
        0,
        0,
    )
    .unwrap();
    let activation = swiglu_launch(0, 8 * 2048 * 4, 8, 2048, correct.packed).unwrap();
    assert!(selected::swiglu_with_prepared_classes(
        &pipelines,
        &gate,
        down,
        activation,
        None,
        8,
        0,
        Some(&classes)
    )
    .is_none());
}
