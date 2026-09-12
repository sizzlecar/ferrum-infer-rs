use super::*;
use crate::backend::metal::vnext_ops::linear::microbench::{weights, Shape};
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{
    AliasPolicy, BlockQuantizationSpec, BufferRequest, BufferUsage, CompositeWeightPart,
    ContractVersion, DeviceId, OperationContract, PhysicalWeightComponentBinding, ProgramValueId,
    QuantizationFormatId, ResolvedStorageComponent, ResolvedTensorSpec, ResolvedValueStorage,
    ResourceId, TensorAccess, WeightComponentRole, WeightId,
};
use half::f16;
use metal::MTLCommandBufferStatus;
use serde_json::json;

fn weight_metadata(gate: bool, composite: bool, format: GgufBlockFormat) -> ResolvedWeightBinding {
    let (hidden, intermediate) = (512_u64, 2048_u64);
    let spec = BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(format.format_id()).unwrap(),
        logical_values_per_block: format.block_values() as u32,
        bytes_per_block: format.block_bytes() as u32,
    };
    let count = if gate && composite { 2 } else { 1 };
    let leaf = |index| PhysicalWeightLayout::BlockQuantized {
        blocks: PhysicalWeightComponentBinding::exact_contiguous(
            WeightId::new(format!("component.{index}")).unwrap(),
        ),
        block_axis: if gate { 2 } else { 1 },
        block_padding: PhysicalWeightPadding::Exact,
    };
    let layout = if gate && composite {
        PhysicalWeightLayout::Composite {
            parts: (0..2)
                .map(|index| CompositeWeightPart {
                    layout: Box::new(leaf(index)),
                    logical_offsets: vec![index, 0, 0],
                    extents: vec![1, intermediate, hidden],
                })
                .collect(),
        }
    } else {
        leaf(0)
    };
    // Resolved physical metadata intentionally exposes validated Deserialize,
    // not constructors that could bypass the core's normal resolution pass.
    serde_json::from_value(json!({
        "weight_id": "weight.fixture", "format_id": "format.fixture",
        "layout_id": "layout.fixture", "schema_version": ContractVersion::new(1, 0),
        "physical_layout": layout,
        "components": (0..count).map(|index| json!({
            "component_id": format!("component.{index}"), "role": WeightComponentRole::PackedValues,
            "physical_dimensions": if gate {
                vec![if composite {1} else {2}, intermediate, hidden / spec.logical_values_per_block as u64]
            } else {vec![hidden, intermediate / spec.logical_values_per_block as u64]},
            "encoding": WeightEncoding::BlockQuantized(spec.clone()),
        })).collect::<Vec<_>>(),
    }))
    .unwrap()
}

#[test]
fn staged_swiglu_workspace_declaration_preserves_layout_and_format_boundaries() {
    let gate = weight_metadata(true, true, GgufBlockFormat::Q4K);
    let down = weight_metadata(false, false, GgufBlockFormat::Q6K);
    // This is the one shared projection, not gate + up + down materializations.
    assert_eq!(
        weight_workspace_bytes(&gate, &down, 512, 2048).unwrap(),
        2 << 20
    );
    let fused = weight_metadata(true, false, GgufBlockFormat::Q4K);
    assert_eq!(weight_workspace_bytes(&fused, &down, 512, 2048).unwrap(), 0);
    let unsupported = weight_metadata(false, false, GgufBlockFormat::Q8_0);
    assert_eq!(
        weight_workspace_bytes(&gate, &unsupported, 512, 2048).unwrap(),
        0
    );
    assert_eq!(weight_workspace_bytes(&gate, &down, 256, 2048).unwrap(), 0);
}

fn gguf_weight_metadata(
    gate: bool,
    hidden: u64,
    intermediate: u64,
    storage: PhysicalStorageLayout,
) -> ResolvedWeightBinding {
    let (name, format, rows, columns) = if gate {
        ("gate_up", GgufBlockFormat::Q4K, intermediate, hidden)
    } else {
        ("down", GgufBlockFormat::Q6K, hidden, intermediate)
    };
    let spec = BlockQuantizationSpec {
        format_id: QuantizationFormatId::new(format.format_id()).unwrap(),
        logical_values_per_block: format.block_values() as u32,
        bytes_per_block: format.block_bytes() as u32,
    };
    let leaf = |index| PhysicalWeightLayout::BlockQuantized {
        blocks: PhysicalWeightComponentBinding {
            component_id: WeightId::new(format!("{name}.{index}")).unwrap(),
            storage: storage.clone(),
        },
        block_axis: if gate { 2 } else { 1 },
        block_padding: PhysicalWeightPadding::Exact,
    };
    let layout = if gate {
        PhysicalWeightLayout::Composite {
            parts: (0..2)
                .map(|index| CompositeWeightPart {
                    layout: Box::new(leaf(index)),
                    logical_offsets: vec![index, 0, 0],
                    extents: vec![1, intermediate, hidden],
                })
                .collect(),
        }
    } else {
        leaf(0)
    };
    // Native GGUF stores each gate/up matrix with rank two. The family schema
    // adds the singleton partition axis through storage strides, not repacking.
    serde_json::from_value(json!({
        "weight_id": name, "format_id": "weight-format.gguf.native-block",
        "layout_id": "layout.gguf.native-block", "schema_version": ContractVersion::new(1, 0),
        "physical_layout": layout,
        "components": (0..if gate {2} else {1}).map(|index| json!({
            "component_id": format!("{name}.{index}"), "role": WeightComponentRole::PackedValues,
            "physical_dimensions": [rows, columns / u64::from(spec.logical_values_per_block)],
            "encoding": WeightEncoding::BlockQuantized(spec.clone()),
        })).collect::<Vec<_>>(),
    }))
    .unwrap()
}

fn resolved_swiglu_bindings(
    gate: ResolvedWeightBinding,
    down: ResolvedWeightBinding,
    hidden: u64,
    intermediate: u64,
    weight_type: ElementType,
) -> Result<Vec<ResolvedValueBinding>, VNextError> {
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
        let element_type = if weight.is_some() {
            weight_type
        } else {
            ElementType::F16
        };
        let tensor =
            ResolvedTensorSpec::new(dimensions, element_type, ResolvedTensorLayout::Contiguous)?;
        let storage = if let Some(weight) = &weight {
            ResolvedValueStorage::composite(
                weight
                    .components()
                    .iter()
                    .map(|component| {
                        ResolvedStorageComponent::new(
                            Some(component.component_id().clone()),
                            ResourceId::new(format!("resource.{}", component.component_id()))?,
                            0,
                            component.physical_bytes()?,
                            component.physical_element_type(),
                        )
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?,
            )?
        } else {
            ResolvedValueStorage::single(
                ResourceId::new(name.clone())?,
                0,
                tensor.minimum_storage_bytes()?,
                element_type,
            )?
        };
        ResolvedValueBinding::new(
            ProgramValueId::new(name)?,
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
    })
    .collect()
}

#[test]
fn staged_swiglu_resolved_gguf_reshape_declares_one_projection_workspace() {
    // Covers the native FFN dimensions and a smaller supported matrix without
    // allocating weights. Both use the same validated physical binding path.
    for (hidden, intermediate) in [(4096, 12288), (512, 2048)] {
        for singleton_stride in [intermediate * (hidden / 256), 1] {
            let gate = gguf_weight_metadata(
                true,
                hidden,
                intermediate,
                PhysicalStorageLayout::Strided {
                    strides_in_elements: vec![singleton_stride, hidden / 256, 1],
                    padding: PhysicalWeightPadding::Exact,
                },
            );
            let down = gguf_weight_metadata(
                false,
                hidden,
                intermediate,
                PhysicalStorageLayout::exact_contiguous(),
            );
            let bindings = resolved_swiglu_bindings(
                gate.clone(),
                down.clone(),
                hidden,
                intermediate,
                ElementType::F16,
            )
            .unwrap();
            dense_swiglu_contract()
                .unwrap()
                .descriptor()
                .validate_resolved_bindings(&bindings)
                .unwrap();
            assert_eq!(
                workspace_bytes(&bindings, hidden, intermediate).unwrap(),
                hidden * intermediate * 2
            );
            let f32_weights =
                resolved_swiglu_bindings(gate, down, hidden, intermediate, ElementType::F32)
                    .unwrap();
            assert_eq!(
                workspace_bytes(&f32_weights, hidden, intermediate).unwrap(),
                0
            );
        }
    }
}

#[test]
fn staged_swiglu_reshape_rejects_noncontiguous_or_padded_storage() {
    let (hidden, intermediate) = (512, 2048);
    for (valid_binding, storage) in [
        (
            false,
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![4096, 3, 1],
                padding: PhysicalWeightPadding::Exact,
            },
        ),
        (
            true,
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![4096, 1, 2048],
                padding: PhysicalWeightPadding::Exact,
            },
        ),
        (
            false,
            PhysicalStorageLayout::Strided {
                strides_in_elements: vec![4096, 2, 1],
                padding: PhysicalWeightPadding::ZeroFill {
                    padded_dimensions: vec![1, 2049, 2],
                },
            },
        ),
        (
            true,
            PhysicalStorageLayout::Tiled {
                tile_shape: vec![1, 8, 2],
                axis_order: vec![0, 1, 2],
                tile_strides_in_elements: vec![4096, 16, 16],
                padding: PhysicalWeightPadding::Exact,
            },
        ),
    ] {
        let gate = gguf_weight_metadata(true, hidden, intermediate, storage);
        let down = gguf_weight_metadata(
            false,
            hidden,
            intermediate,
            PhysicalStorageLayout::exact_contiguous(),
        );
        let bindings = resolved_swiglu_bindings(gate, down, hidden, intermediate, ElementType::F16);
        // Holes/padding exceed the declared physical source span. Transposed
        // and tiled bindings are valid, but this row-major kernel rejects them.
        if valid_binding {
            let bindings = bindings.unwrap();
            assert_eq!(workspace_bytes(&bindings, hidden, intermediate).unwrap(), 0);
        } else {
            assert!(bindings.is_err());
        }
    }
}

pub(super) fn region<T: Copy>(
    runtime: &MetalDeviceRuntime,
    name: &str,
    values: &[T],
    element_type: ElementType,
) -> MetalBufferRegion {
    let region = runtime
        .allocate_test_region(
            &BufferRequest::new(
                ResourceId::new(name).unwrap(),
                std::mem::size_of_val(values) as u64,
                64,
                BufferUsage::Transfer,
                element_type,
            )
            .unwrap(),
        )
        .unwrap();
    overwrite(&region, values);
    region
}

pub(super) fn overwrite<T: Copy>(region: &MetalBufferRegion, values: &[T]) {
    assert_eq!(
        region.length_bytes() as usize,
        std::mem::size_of_val(values)
    );
    // SAFETY: the fixture owns this live shared allocation and has waited for
    // every preceding command before accessing it. The exact span is checked.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            std::mem::size_of_val(values),
        );
    }
}

pub(super) fn bytes(region: &MetalBufferRegion) -> Vec<u8> {
    // SAFETY: called only after completion, over the retained allocation span.
    unsafe {
        std::slice::from_raw_parts(
            region
                .buffer()
                .contents()
                .cast::<u8>()
                .add(region.offset_bytes() as usize),
            region.length_bytes() as usize,
        )
        .to_vec()
    }
}

fn sequence_weights(shape: Shape) -> Vec<u8> {
    let mut values = weights(shape);
    // The single-projection fixture deliberately uses large coefficients.
    // A complete gate * up -> down sequence needs smaller scales to keep its
    // F16 intermediate and output finite, including the unchanged baseline.
    let scale_offsets: &[usize] = match shape.format {
        GgufBlockFormat::Q4K => &[0, 2],
        GgufBlockFormat::Q6K => &[208],
        _ => unreachable!("this sequence covers Q4_K gate/up and Q6_K down"),
    };
    for block in values.chunks_exact_mut(shape.format.block_bytes()) {
        for &offset in scale_offsets {
            let scale = f16::from_le_bytes(block[offset..offset + 2].try_into().unwrap());
            block[offset..offset + 2]
                .copy_from_slice(&f16::from_f32(scale.to_f32() / 32.0).to_le_bytes());
        }
    }
    values
}

#[test]
fn staged_swiglu_sequence_preserves_thresholds_guards_and_workspace_reuse() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("device.swiglu.staging").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalLinearPipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    let (hidden, intermediate) = (512_u64, 2048_u64);
    let gate_values = sequence_weights(Shape {
        name: "gate",
        input: hidden as u32,
        output: intermediate as u32,
        format: GgufBlockFormat::Q4K,
    });
    let mut up_values = gate_values.clone();
    up_values.rotate_left(GgufBlockFormat::Q4K.block_bytes());
    assert_ne!(gate_values, up_values);
    let down_values = sequence_weights(Shape {
        name: "down",
        input: intermediate as u32,
        output: hidden as u32,
        format: GgufBlockFormat::Q6K,
    });
    let guard = f16::from_f32(123.0);
    let staging_bytes = hidden * intermediate * 2;

    for (rows, staged_dispatches) in [(255_u64, 0), (256, 1), (767, 1), (768, 3)] {
        let mut input = vec![guard; 3];
        input.extend((0..rows * hidden).map(|i| f16::from_f32((i as f32 * 0.013).sin() * 0.125)));
        input.extend([guard; 5]);
        let mut output = vec![guard; (rows * hidden) as usize + 12];
        output[5..5 + (rows * hidden) as usize].fill(f16::NAN);
        let activation_offset = 64 + rows * intermediate * 4;
        let staging_offset = 64 + rows * intermediate * 6;
        let mut scratch = vec![guard; ((staging_offset + staging_bytes + 64) / 2) as usize];
        scratch[32..((staging_offset + staging_bytes) / 2) as usize].fill(f16::NAN);
        let regions = vec![
            region(runtime, "input", &input, ElementType::F16),
            region(runtime, "gate", &gate_values, ElementType::U8),
            region(runtime, "up", &up_values, ElementType::U8),
            region(runtime, "down", &down_values, ElementType::U8),
            region(runtime, "output", &output, ElementType::F16),
            region(runtime, "scratch", &scratch, ElementType::U8),
        ];
        let source_before = regions[..4].iter().map(bytes).collect::<Vec<_>>();
        let gate_up = [1, 2]
            .into_iter()
            .enumerate()
            .map(|(index, weight_region)| {
                linear_launch(
                    PreparedLinearPart {
                        region: weight_region,
                        format: LinearPhysicalFormat::Q4K,
                        output_offset: index as u32 * intermediate as u32,
                        out_features: intermediate as u32,
                    },
                    0,
                    5,
                    rows,
                    hidden,
                    2 * intermediate,
                    6,
                    64,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let down = linear_launch(
            PreparedLinearPart {
                region: 3,
                format: LinearPhysicalFormat::Q6K,
                output_offset: 0,
                out_features: hidden as u32,
            },
            5,
            4,
            rows,
            intermediate,
            hidden,
            activation_offset,
            10,
        )
        .unwrap();
        validate_launch_regions_with_raw_workspace(&regions, &gate_up, &[5]).unwrap();
        validate_launch_regions_with_raw_workspace(&regions, &[down], &[5]).unwrap();
        let workspace = Workspace::new(
            &regions,
            5,
            staging_offset,
            staging_bytes,
            gate_up.iter().copied().chain([down]),
        )
        .unwrap();
        if rows == 768 {
            let untouched = bytes(&regions[4]);
            assert!(Workspace::new(
                &regions,
                5,
                staging_offset,
                staging_bytes - 2,
                gate_up.iter().copied().chain([down])
            )
            .is_err());
            assert!(Workspace::new(
                &regions,
                5,
                activation_offset,
                staging_bytes,
                gate_up.iter().copied().chain([down])
            )
            .is_err());
            assert!(Workspace::new(
                &regions,
                5,
                staging_offset,
                staging_bytes + 66,
                gate_up.iter().copied().chain([down])
            )
            .is_err());
            assert_eq!(
                bytes(&regions[4]),
                untouched,
                "rejected plans submit no work"
            );
        }
        let mut sequence = Sequence {
            gate_up,
            down,
            activation: swiglu_launch(64, activation_offset, rows, intermediate, 2 * intermediate)
                .unwrap(),
            scratch_region: 5,
            workspace: None,
        };
        let mut reference = None;
        for candidate in [false, true] {
            overwrite(&regions[4], &output);
            overwrite(&regions[5], &scratch);
            sequence.workspace = if candidate { workspace } else { None };
            assert_eq!(
                sequence.dispatch_count(),
                4 + if candidate { staged_dispatches } else { 0 }
            );
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            sequence.encode(&pipelines, encoder, &regions);
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            let actual = bytes(&regions[4]);
            for (index, value) in actual.chunks_exact(2).enumerate() {
                let value = f16::from_le_bytes(value.try_into().unwrap());
                if (5..5 + (rows * hidden) as usize).contains(&index) {
                    assert!(
                        value.is_finite(),
                        "rows={rows} candidate={candidate} output {index}: {value:?}"
                    );
                } else {
                    assert_eq!(value, guard, "output guard {index}");
                }
            }
            if let Some(reference) = &reference {
                assert_eq!(
                    &actual, reference,
                    "rows={rows} complete production sequence output"
                );
            } else {
                reference = Some(actual);
            }
            let scratch_after = bytes(&regions[5]);
            assert!(scratch_after[..64]
                .chunks_exact(2)
                .all(|pair| pair == guard.to_le_bytes()));
            assert!(scratch_after[scratch_after.len() - 64..]
                .chunks_exact(2)
                .all(|pair| pair == guard.to_le_bytes()));
            let stage =
                &scratch_after[staging_offset as usize..(staging_offset + staging_bytes) as usize];
            assert!(stage.chunks_exact(2).all(|pair| {
                let value = f16::from_le_bytes(pair.try_into().unwrap());
                if candidate && staged_dispatches > 0 {
                    value.is_finite()
                } else {
                    value.is_nan()
                }
            }));
        }
        assert_eq!(
            regions[..4].iter().map(bytes).collect::<Vec<_>>(),
            source_before
        );
    }
}
