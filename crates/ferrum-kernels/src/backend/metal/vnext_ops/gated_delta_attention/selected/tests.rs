use super::super::super::linear::prepare_leaf_encoding;
use super::super::super::weights::{component_metadata, resolve_layout};
use super::*;
use ferrum_interfaces::vnext::{
    ContractVersion, PhysicalWeightLayout, ResolvedWeightBinding, WeightComponentRole,
    WeightComponentSpec, WeightEncoding, WeightSchema, WeightTensorSpec,
};

fn shape() -> AttentionShape {
    AttentionShape {
        hidden_size: 256,
        key_heads: 2,
        value_heads: 4,
        key_dim: 64,
        value_dim: 64,
        qkv_features: 512,
        value_features: 256,
        qkvz_features: 768,
        ba_features: 8,
        qkvzba_features: 776,
        conv_kernel: 4,
        conv_state_width: 3,
        epsilon: 1e-6,
        layer_index: 0,
        decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
        value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
    }
}
fn dense(output: u64, input: u64) -> super::super::super::linear::PreparedLinearPart {
    let component: ferrum_interfaces::vnext::WeightId =
        "component.dense".to_owned().try_into().unwrap();
    let weight_id: ferrum_interfaces::vnext::WeightId =
        "weight.dense".to_owned().try_into().unwrap();
    let schema = WeightSchema {
        format_id: "weight-format.dense".to_owned().try_into().unwrap(),
        layout_id: "weight-layout.test-dense".to_owned().try_into().unwrap(),
        version: ContractVersion::new(1, 0),
        components: vec![WeightComponentSpec {
            id: component.clone(),
            role: WeightComponentRole::Values,
            external_names: vec!["dense".into()],
            dimensions: vec![output, input],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        }],
        tensors: vec![WeightTensorSpec {
            id: weight_id.clone(),
            dimensions: vec![output, input],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: component,
            },
            required: true,
        }],
    };
    let weight = ResolvedWeightBinding::from_schema(&schema, &weight_id).unwrap();
    prepare_leaf_encoding(
        &[component_metadata(&weight.components()[0])],
        &resolve_layout(&weight).unwrap(),
        output,
        input,
        1,
        0,
    )
    .unwrap()
}
fn projections(shape: AttentionShape, tokens: u64) -> (Vec<LinearLaunch>, LinearLaunch) {
    let make = |out, width, stride| {
        linear_launch(dense(out, width), 0, 0, tokens, width, stride, 0, 0).unwrap()
    };
    (
        vec![make(
            shape.qkvzba_features,
            shape.hidden_size,
            shape.qkvzba_features,
        )],
        make(shape.hidden_size, shape.value_features, shape.hidden_size),
    )
}
#[test]
fn recurrent_gdn_selected_chain_preserves_packed_order_work_and_unknown_boundary() {
    let device = Device::system_default().expect("actual Metal GDN PSO catalog");
    let a = MetalGatedDeltaPipelines::new(&device).unwrap();
    let l = MetalLinearPipelines::new(&device).unwrap();
    let p = MetalPrimitivePipelines::new(&device).unwrap();
    let s = shape();
    let (i1, o1) = projections(s, 1);
    let (i3, o3) = projections(s, 3);
    let (ip, op) = projections(s, 4);
    let rows = [
        Row {
            projection: Projection {
                params: s.params(1).unwrap(),
                input: &i1,
                output: o1,
                staged: false,
            },
            form: GatedDeltaExecutionForm::RecurrentScan,
        },
        Row {
            projection: Projection {
                params: s.params(3).unwrap(),
                input: &i3,
                output: o3,
                staged: false,
            },
            form: GatedDeltaExecutionForm::RecurrentScan,
        },
    ];
    let packed = Projection {
        params: s.params(4).unwrap(),
        input: &ip,
        output: op,
        staged: false,
    };
    let scratch = ScratchLayout::new(s, 4).unwrap().required_bytes;
    let packed_e = evidence(
        &a,
        &l,
        &p,
        ElementType::F32,
        4,
        scratch,
        Some(packed),
        rows.into_iter(),
    )
    .unwrap();
    let serial_e = evidence(
        &a,
        &l,
        &p,
        ElementType::F32,
        4,
        scratch,
        None,
        rows.into_iter(),
    )
    .unwrap();
    let packed_n =
        5 + ip.iter().map(|v| v.dispatch_count()).sum::<u64>() + op.dispatch_count() + 2 * 4;
    let serial_n = rows
        .iter()
        .map(|r| {
            9 + r
                .projection
                .input
                .iter()
                .map(|v| v.dispatch_count())
                .sum::<u64>()
                + r.projection.output.dispatch_count()
        })
        .sum();
    packed_e.validate_command(4, packed_n, 0).unwrap();
    serial_e.validate_command(4, serial_n, 0).unwrap();
    assert_ne!(packed_e.family_signature(), serial_e.family_signature());
    assert_eq!(packed_e.work().peak_scratch_bytes, scratch);
    assert!(packed_e.work().inner_work_units > 4 * 256);
    assert!(evidence(
        &a,
        &l,
        &p,
        ElementType::F32,
        5,
        scratch,
        Some(packed),
        rows.into_iter()
    )
    .is_none());
    // Recurrent selector is the same actual PSO used by execution.
    let (actual, entry, _, _) = recurrent(&a, &rows[0].projection.params);
    assert!(std::ptr::eq(actual, &a.simd_delta) || std::ptr::eq(actual, &a.delta));
    assert_eq!(
        entry,
        if std::ptr::eq(actual, &a.simd_delta) {
            SIMD_DELTA_KERNEL
        } else {
            DELTA_KERNEL
        }
    );
    let chunked = GatedDeltaExecutionCapabilities::with_chunked_scan(64)
        .unwrap()
        .select(64, GatedDeltaExecutionPreference::ChunkedScan)
        .unwrap();
    let mut unknown = rows;
    unknown[0].form = chunked;
    assert!(evidence(
        &a,
        &l,
        &p,
        ElementType::F32,
        4,
        scratch,
        Some(packed),
        unknown.into_iter()
    )
    .is_none());
}
