use super::*;
use ferrum_quantization::SafetensorsArchive;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn fixture(
    directory: &std::path::Path,
) -> (ComfyCanonicalWeightSource, ModelProgram, WeightSchema) {
    let weight = [1.00390625f32, 1.01171875, -1.00390625, -0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let bias = [0.0f32, -1.01171875]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let buffer = [0.12345679f32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "weight",
                TensorView::new(Dtype::F32, vec![2, 2], &weight).unwrap(),
            ),
            ("bias", TensorView::new(Dtype::F32, vec![2], &bias).unwrap()),
            (
                "buffer",
                TensorView::new(Dtype::F32, vec![1], &buffer).unwrap(),
            ),
        ],
        &None,
        &directory.join("tiny.safetensors"),
    )
    .unwrap();
    let source = ComfyCanonicalWeightSource::new(
        SafetensorsArchive::open_single_file(directory, "tiny.safetensors").unwrap(),
    )
    .unwrap();
    let mut components = vec![];
    let mut tensors = vec![];
    let mut weights = vec![];
    for (name, dimensions) in [
        ("weight", vec![2, 2]),
        ("bias", vec![2]),
        ("buffer", vec![1]),
    ] {
        let component_id = WeightId::new(format!("component.{name}")).unwrap();
        let weight_id = WeightId::new(format!("weight.{name}")).unwrap();
        components.push(WeightComponentSpec {
            id: component_id.clone(),
            role: WeightComponentRole::Values,
            external_names: vec![name.into()],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F32,
            },
            required: true,
        });
        tensors.push(WeightTensorSpec {
            id: weight_id.clone(),
            dimensions: dimensions.clone(),
            logical_element_type: ElementType::F32,
            physical_layout: PhysicalWeightLayout::Dense { component_id },
            required: true,
        });
        weights.push(WeightReference {
            weight_id,
            value_id: ProgramValueId::new(format!("value.{name}")).unwrap(),
            tensor: ProgramTensorSpec {
                dimensions,
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
    }
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.fixture").unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.fixture").unwrap(),
        version: ContractVersion::new(1, 0),
        components,
        tensors,
    };
    let program = ModelProgram::new(
        ModelFamilyId::new("family.source.fixture").unwrap(),
        vec![ProgramValueId::new("value.input").unwrap()],
        vec![ProgramBlock {
            id: "block.fixture".into(),
            nodes: vec![ProgramNode {
                id: NodeId::new("node.affine").unwrap(),
                operation_id: OperationId::new(AFFINE_LINEAR_F32_OPERATION_ID).unwrap(),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: ["value.input", "value.weight", "value.bias"]
                    .map(|s| ProgramValueId::new(s).unwrap())
                    .to_vec(),
                outputs: vec![ProgramValueId::new("value.output").unwrap()],
                attributes: BTreeMap::new(),
            }],
        }],
        vec![],
        weights,
        vec![ProgramValueId::new("value.output").unwrap()],
    )
    .unwrap();
    (source, program, schema)
}

#[test]
fn selection_is_operand_based_and_rejects_alias_uses_outside_affine() {
    let directory = tempfile::tempdir().unwrap();
    let (source, program, schema) = fixture(directory.path());
    let selected = select(&program, &schema, &source, 8).unwrap();
    assert_eq!(
        selected
            .components
            .keys()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        ["component.bias", "component.weight"]
    );
    assert_eq!(selected.chunk_bytes, 8);
    let mut blocks = program.blocks().to_vec();
    let mut extra = blocks[0].nodes[0].clone();
    extra.id = NodeId::new("node.other_use").unwrap();
    extra.operation_id = OperationId::new(DENSE_LINEAR_F32_OPERATION_ID).unwrap();
    extra.inputs.pop();
    extra.outputs = vec![ProgramValueId::new("value.other_output").unwrap()];
    blocks[0].nodes.push(extra);
    let wrong = ModelProgram::new(
        program.family_id().clone(),
        program.inputs().to_vec(),
        blocks,
        vec![],
        program.weights().to_vec(),
        program.outputs().to_vec(),
    )
    .unwrap();
    assert!(select(&wrong, &schema, &source, 8).is_err());
    assert!(select(&program, &schema, &source, 3).is_err());
}

#[test]
fn roundtrip_uses_rne_and_preserves_zero_subnormal_and_finite_boundary() {
    let input = [
        1.00390625f32,
        1.01171875,
        -1.00390625,
        -1.01171875,
        0.0,
        -0.0,
        f32::from_bits(0x00010000),
        f32::from_bits(1),
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect::<Vec<_>>();
    let mut output = vec![0; input.len()];
    round_into(&input, &mut output).unwrap();
    let actual = output
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        [0x3f800000, 0x3f820000, 0xbf800000, 0xbf820000, 0, 0x80000000, 0x00010000, 0]
    );
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(round_into(&bad.to_le_bytes(), &mut [0; 4]).is_err());
    }
    assert!(round_into(&[0; 3], &mut [0; 3]).is_err());
}

#[test]
fn source_ranges_charge_original_host_reject_bypass_and_refund_after_last_owner() {
    let directory = tempfile::tempdir().unwrap();
    let (original, program, schema) = fixture(directory.path());
    let selected = select(&program, &schema, &original, 8).unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            8,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: 8,
            },
        )
        .unwrap(),
    );
    // Only the pure range/owner layer is exercised here; this is not a forged
    // StaticWeightSourceContext nor evidence that a GPU upload occurred.
    let source = RoundtripSource {
        original: &original,
        selection: &selected,
        host: host.clone(),
        plan_hash: serde_json::from_value(json!("a".repeat(64))).unwrap(),
        family_fingerprint: "fixture-family".into(),
        ready: AtomicBool::new(true),
        failed: AtomicBool::new(false),
        progress: Mutex::new(BTreeMap::new()),
    };
    let weight = selected
        .components
        .get(&WeightId::new("component.weight").unwrap())
        .unwrap();
    assert!(source.component(weight.component()).is_err());
    assert!(source.component_segments(weight.component()).is_err());
    let buffer = schema
        .components
        .iter()
        .find(|c| c.external_names == ["buffer"])
        .unwrap();
    assert_eq!(
        source.component(buffer).unwrap().bytes(),
        original.archive().tensor("buffer").unwrap().bytes()
    );
    let first = source.materialize(weight, 0, 8).unwrap();
    assert!(host.owns(&first));
    assert_eq!(first.class(), HostMemoryClass::WeightBacking);
    assert_eq!(
        first.bytes(),
        &[1.0f32.to_le_bytes(), 1.015625f32.to_le_bytes()].concat()
    );
    let alias = first.clone();
    drop(first);
    assert_eq!(host.usage().unwrap().committed_bytes(), 8);
    assert!(host
        .allocate_zeroed(HostMemoryClass::WeightBacking, 1)
        .is_err());
    drop(alias);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let second = source.materialize(weight, 8, 8).unwrap();
    drop(second);
    let bias = selected
        .components
        .get(&WeightId::new("component.bias").unwrap())
        .unwrap();
    drop(source.materialize(bias, 0, 8).unwrap());
    let receipt = source.receipt().unwrap();
    assert_eq!(receipt["components"].as_array().unwrap().len(), 2);
    assert_ne!(
        receipt["components"][0]["original_f32_sha256"],
        receipt["components"][0]["bf16_roundtrip_f32_sha256"]
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(source.materialize(weight, 8, 8).is_err());
    assert!(source.receipt().is_err());
}
