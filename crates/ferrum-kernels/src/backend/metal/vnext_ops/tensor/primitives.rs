//! Shared semantic validation and scalar/layout primitives; no model identities.
use super::*;
use ferrum_runtime::{PhysicalWeightLayout, TensorPrimitiveBf16Geometry};

pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<TensorPrimitiveBf16Geometry, String> {
    let kind = kind.primitive().ok_or("not a BF16 tensor primitive")?;
    let g =
        TensorPrimitiveBf16Geometry::new(kind, values, attributes).map_err(|e| e.to_string())?;
    for value in values {
        if let Some(weight) = value.weight() {
            weight
                .validate_logical(value.tensor().dimensions(), value.tensor().element_type())
                .map_err(|e| e.to_string())?;
            if !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. }) {
                return Err("Metal tensor primitive requires dense physical storage".into());
            }
        }
    }
    Ok(g)
}

#[repr(C)]
struct Params {
    elements: u64,
    grid_width: u64,
    columns: u64,
    head_dim: u64,
    kind: u32,
    reserved: u32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 40);

pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let g = geometry(kind, participant.bindings(), participant.attributes())?;
    let (mode, per_element) = match kind {
        // Logical arithmetic treats erf as one transcendental, like tanh in
        // the existing GELU contract; it is not a hardware instruction count.
        MetalTensorOperation::GeluErf => (0, 5_u64),
        MetalTensorOperation::HalfRotary => (1, 3),
        MetalTensorOperation::ReshapeCopy => (2, 0),
        _ => return Err("not an elementwise primitive".into()),
    };
    let mut regions = Vec::new();
    for (port, &bytes) in g.input_bytes.iter().enumerate() {
        regions.push(tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port as u32,
            if mode == 1 && port > 0 {
                ElementType::F32
            } else {
                ElementType::Bf16
            },
            bytes,
        )?);
    }
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        g.output_bytes,
    )?;
    for input in &regions {
        disjoint(input, &output)?;
    }
    let output_index = regions.len();
    regions.push(output);
    let params = Params {
        elements: g.elements(),
        grid_width: g.elements().min(1 << 20),
        columns: g.columns,
        head_dim: g.head_dim,
        kind: mode,
        reserved: 0,
    };
    let operations = g
        .elements()
        .checked_mul(per_element)
        .ok_or("Metal primitive work overflows")?;
    let work =
        DeviceCommandLogicalWork::tensor(g.elements(), operations).map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.tensor.primitive.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            let pipeline = pipelines
                .primitives
                .as_ref()
                .expect("selected primitive pipeline");
            compute.set_compute_pipeline_state(pipeline);
            set_region(compute, 0, &regions[0]);
            // Non-rotary kernels never dereference either table placeholder.
            set_region(compute, 1, &regions[if mode == 1 { 1 } else { 0 }]);
            set_region(compute, 2, &regions[if mode == 1 { 2 } else { 0 }]);
            set_region(compute, 3, &regions[output_index]);
            set_params(compute, 4, &params);
            compute.dispatch_threads(
                MTLSize::new(params.grid_width, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::super::tests::value;
    use super::*;
    use ferrum_runtime::CanonicalRational;

    #[test]
    fn original_patch_rank_and_rotary_table_type_are_required() {
        let inputs = |weight: &[u64]| {
            vec![
                value(
                    "x",
                    ResolvedValueRole::Input,
                    0,
                    &[3, 24],
                    ElementType::Bf16,
                ),
                value("w", ResolvedValueRole::Input, 1, weight, ElementType::Bf16),
                value("b", ResolvedValueRole::Input, 2, &[7], ElementType::Bf16),
                value(
                    "y",
                    ResolvedValueRole::Output,
                    0,
                    &[3, 7],
                    ElementType::Bf16,
                ),
            ]
        };
        assert!(geometry(
            MetalTensorOperation::PatchProjection,
            &inputs(&[7, 3, 2, 2, 2]),
            &BTreeMap::new()
        )
        .is_ok());
        assert!(geometry(
            MetalTensorOperation::PatchProjection,
            &inputs(&[7, 24]),
            &BTreeMap::new()
        )
        .is_err());
        let attrs = BTreeMap::from([(
            AttributeId::new("head_dim").unwrap(),
            SemanticValue::Unsigned(72),
        )]);
        for dtype in [ElementType::F32, ElementType::Bf16] {
            let values = vec![
                value(
                    "x",
                    ResolvedValueRole::Input,
                    0,
                    &[3, 144],
                    ElementType::Bf16,
                ),
                value("c", ResolvedValueRole::Input, 1, &[3, 72], dtype),
                value("s", ResolvedValueRole::Input, 2, &[3, 72], dtype),
                value(
                    "y",
                    ResolvedValueRole::Output,
                    0,
                    &[3, 144],
                    ElementType::Bf16,
                ),
            ];
            assert_eq!(
                geometry(MetalTensorOperation::HalfRotary, &values, &attrs).is_ok(),
                dtype == ElementType::F32
            );
        }
    }

    #[test]
    fn affine_norm_admits_real_merger_width_at_exact_device_boundary() {
        let width = 4608;
        let values = vec![
            value(
                "x",
                ResolvedValueRole::Input,
                0,
                &[3, width],
                ElementType::Bf16,
            ),
            value(
                "g",
                ResolvedValueRole::Input,
                1,
                &[width],
                ElementType::Bf16,
            ),
            value(
                "b",
                ResolvedValueRole::Input,
                2,
                &[width],
                ElementType::Bf16,
            ),
            value(
                "y",
                ResolvedValueRole::Output,
                0,
                &[3, width],
                ElementType::Bf16,
            ),
        ];
        let attrs = BTreeMap::from([
            (
                AttributeId::new("hidden_size").unwrap(),
                SemanticValue::Unsigned(width),
            ),
            (
                AttributeId::new("epsilon").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 1_000_000).unwrap()),
            ),
        ]);
        assert!(norm::geometry(
            MetalTensorOperation::AffineLayerNorm,
            &values,
            &attrs,
            32768
        )
        .is_ok());
        assert!(norm::geometry(
            MetalTensorOperation::AffineLayerNorm,
            &values,
            &attrs,
            32767
        )
        .is_err());
    }

    #[test]
    fn isolated_attention_requires_complete_partition_and_admitted_on_chip_state() {
        let mut values: Vec<_> = (0..3)
            .map(|port| {
                value(
                    &format!("x{port}"),
                    ResolvedValueRole::Input,
                    port,
                    &[35, 144],
                    ElementType::Bf16,
                )
            })
            .collect();
        values.push(value(
            "y",
            ResolvedValueRole::Output,
            0,
            &[35, 144],
            ElementType::Bf16,
        ));
        let mut attrs = BTreeMap::from([
            (
                AttributeId::new("head_dim").unwrap(),
                SemanticValue::Unsigned(72),
            ),
            (
                AttributeId::new("scale").unwrap(),
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            ),
            (
                AttributeId::new("segment_lengths").unwrap(),
                SemanticValue::Integers(vec![3, 31, 1]),
            ),
        ]);
        let required = (2048 + 16 * 72 + 96) * 4;
        assert!(attention::geometry(
            MetalTensorOperation::SegmentedAttention,
            &values,
            &attrs,
            required
        )
        .is_ok());
        assert!(attention::geometry(
            MetalTensorOperation::SegmentedAttention,
            &values,
            &attrs,
            required - 1
        )
        .is_err());
        attrs.insert(
            AttributeId::new("segment_lengths").unwrap(),
            SemanticValue::Integers(vec![3, 31]),
        );
        assert!(attention::geometry(
            MetalTensorOperation::SegmentedAttention,
            &values,
            &attrs,
            required
        )
        .is_err());
    }
}
