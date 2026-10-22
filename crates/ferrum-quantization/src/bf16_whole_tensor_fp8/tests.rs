use super::*;
use crate::SafetensorsArchive;
use half::bf16;
use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn source_spec(name: &str) -> WeightComponentSpec {
    WeightComponentSpec {
        id: id(format!("component.{name}")),
        role: WeightComponentRole::Values,
        external_names: vec![format!("{name}.weight")],
        dimensions: vec![1, 4],
        required: true,
        encoding: WeightEncoding::Dense {
            element_type: ElementType::Bf16,
        },
    }
}
fn source(
    values: &[[u16; 4]],
    dtype: Dtype,
) -> (
    tempfile::TempDir,
    SafetensorsArchive,
    Vec<WeightComponentSpec>,
) {
    let directory = tempfile::tempdir().unwrap();
    let specs: Vec<_> = (0..values.len())
        .map(|i| source_spec(&format!("part{i}")))
        .collect();
    let bytes: Vec<Vec<_>> = values
        .iter()
        .map(|row| row.iter().flat_map(|x| x.to_le_bytes()).collect())
        .collect();
    let tensors = specs
        .iter()
        .zip(&bytes)
        .map(|(spec, bytes)| {
            (
                spec.external_names[0].as_str(),
                TensorView::new(dtype, vec![1, 4], bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(tensors, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, archive, specs)
}

#[test]
fn integer_encoder_matches_independent_f64_enumeration_for_all_finite_bf16_inputs() {
    fn independent_decode(bits: u8) -> f64 {
        let exponent = (bits >> 3) & 15;
        if exponent == 0 {
            f64::from(bits & 7) / 512.0
        } else {
            (1.0 + f64::from(bits & 7) / 8.0) * 2_f64.powi(i32::from(exponent) - 7)
        }
    }
    for bits in 0..=u16::MAX {
        let value = bf16::from_bits(bits).to_f32();
        if !value.is_finite() {
            continue;
        }
        let target = f64::from(value.abs());
        let mut best = 0_u8;
        let mut distance = f64::INFINITY;
        for candidate in 0..=126_u8 {
            let error = (independent_decode(candidate) - target).abs();
            if error < distance || (error == distance && candidate & 1 == 0) {
                best = candidate;
                distance = error;
            }
        }
        // Above the endpoint all finite values saturate. Using subtraction
        // for enormous magnitudes loses distances, so state that endpoint first.
        if target >= 448.0 {
            best = 126;
        }
        let expected = best | if value.is_sign_negative() { 128 } else { 0 };
        assert_eq!(conversion::encode(value), expected, "BF16 input {bits:04x}");
    }
    assert_eq!(conversion::encode(1.0625), 0x38);
    assert_eq!(conversion::encode(1.1875), 0x3a);
    assert_eq!(conversion::encode(1.0 / 1024.0), 0);
    assert_eq!(conversion::encode(-0.0), 128);
}

#[test]
fn f32_midpoints_and_neighbors_use_even_fp8_significands() {
    // All finite positive FP8 neighbors have exactly representable F32
    // midpoints. Static reciprocal multiplication can produce the adjoining
    // F32 values even when its input was BF16.
    for lower in 0..126_u8 {
        let midpoint = (conversion::decode(lower) + conversion::decode(lower + 1)) * 0.5;
        let below = f32::from_bits(midpoint.to_bits() - 1);
        let above = f32::from_bits(midpoint.to_bits() + 1);
        assert_eq!(conversion::encode(below), lower);
        assert_eq!(conversion::encode(above), lower + 1);
        let even = if lower & 1 == 0 { lower } else { lower + 1 };
        assert_eq!(conversion::encode(midpoint), even);
        assert_eq!(conversion::encode(-midpoint), even | 128);
    }
}

#[test]
fn shared_amax_real_mmap_produces_one_admitted_backing_with_retained_slices() {
    let (_directory, archive, specs) = source(
        &[
            [0x43e0, 0x3f88, 0xbf98, 0x8000], //448,1.0625,-1.1875,-0
            [0x4360, 0x3f88, 0xbf98, 0],      //224, same midpoint values
            [0x42e0, 0x3f80, 0xbf80, 0x3b00], //112,1,-1,1/512
        ],
        Dtype::BF16,
    );
    let refs: Vec<_> = specs.iter().collect();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs: Vec<_> = derived.iter().collect();
    let host = account(24);
    let payloads = conversion::materialize(&host, &archive, &refs, &outputs).unwrap();
    assert_eq!(payloads[0].bytes(), &[126, 56, 186, 128]);
    assert_eq!(payloads[1].bytes(), &[118, 56, 186, 0]);
    assert_eq!(payloads[2].bytes(), &[110, 56, 184, 1]);
    assert_eq!(payloads.len(), 6);
    for payload in &payloads[3..] {
        assert_eq!(payload.bytes(), &1_f32.to_le_bytes());
    }
    assert_eq!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::WeightBacking),
        24
    );
    let retained = payloads[1].retained_host_memory().unwrap().clone();
    drop(payloads);
    drop(archive);
    assert_eq!(retained.bytes(), &[118, 56, 186, 0]);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn capacity_failure_wrong_recipe_and_nonfinite_leave_no_charge() {
    let (_directory, archive, specs) = source(
        &[[0x43e0, 0x3f80, 0, 0], [0x4360, 0x3f80, 0, 0]],
        Dtype::BF16,
    );
    let refs: Vec<_> = specs.iter().collect();
    let derived = plan::derive_group(&refs).unwrap();
    let outputs: Vec<_> = derived.iter().collect();
    let host = account(15); //8 packed bytes +2*4 scale bytes are required atomically.
    assert!(conversion::materialize(&host, &archive, &refs, &outputs).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let host = account(16);
    assert!(conversion::materialize(&host, &archive, &[refs[1], refs[0]], &outputs).is_err());
    assert!(conversion::materialize(&host, &archive, &refs, &[outputs[0], outputs[0]]).is_err());
    let mut wrong = derived[0].clone();
    wrong.dimensions = vec![2, 2];
    assert!(conversion::materialize(&host, &archive, &refs, &[&wrong]).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    for value in [0x7fc0, 0x7f80, 0xff80, 0x0001] {
        let (_directory, archive, specs) = source(&[[value, 0, 0, 0]], Dtype::BF16);
        let refs = vec![&specs[0]];
        let derived = plan::derive_group(&refs).unwrap();
        assert!(conversion::materialize(
            &host,
            &archive,
            &refs,
            &derived.iter().collect::<Vec<_>>()
        )
        .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn zero_group_has_scale_one_and_implicit_source_transcoding_is_rejected() {
    let host = account(8);
    let (_directory, archive, specs) = source(&[[0, 0x8000, 0, 0]], Dtype::BF16);
    let refs = vec![&specs[0]];
    let derived = plan::derive_group(&refs).unwrap();
    let outputs = derived.iter().collect::<Vec<_>>();
    let payloads = conversion::materialize(&host, &archive, &refs, &outputs).unwrap();
    assert_eq!(payloads[0].bytes(), &[0, 128, 0, 0]);
    assert_eq!(payloads[1].bytes(), &1_f32.to_le_bytes());
    drop(payloads);
    let (_directory, archive, specs) = source(&[[0, 0, 0, 0]], Dtype::F16);
    let exact = ExactBf16SafetensorsSource::new(&archive);
    assert!(conversion::materialize(&host, &exact, &[&specs[0]], &outputs).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn unchanged_component_retains_source_without_a_converted_copy() {
    let (_directory, archive, specs) = source(&[[0x43e0, 0x3f80, 0xbf80, 0]], Dtype::BF16);
    let host = account(8);
    let outputs = vec![&specs[0]];
    let payloads = conversion::materialize(&host, &archive, &[&specs[0]], &outputs).unwrap();
    assert_eq!(
        payloads[0].bytes(),
        archive.tensor("part0.weight").unwrap().bytes()
    );
    assert!(payloads[0].retained_host_memory().is_some());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(payloads);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

fn fixture(operation: &str) -> (WeightSchema, ModelProgram, Vec<Vec<WeightId>>) {
    let components = vec![
        source_spec("q"),
        source_spec("k"),
        source_spec("v"),
        source_spec("untouched"),
    ];
    let tensors: Vec<_> = components
        .iter()
        .map(|component| WeightTensorSpec {
            id: id(format!("weight.{}", component.external_names[0])),
            dimensions: component.dimensions.clone(),
            logical_element_type: ElementType::Bf16,
            required: true,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: component.id.clone(),
            },
        })
        .collect();
    let refs: Vec<_> = tensors
        .iter()
        .enumerate()
        .map(|(i, t)| WeightReference {
            weight_id: t.id.clone(),
            value_id: id(format!("value.weight{i}")),
            tensor: ProgramTensorSpec {
                dimensions: t.dimensions.clone(),
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        })
        .collect();
    let nodes = refs
        .iter()
        .enumerate()
        .map(|(i, r)| ProgramNode {
            id: id(format!("node.{i}")),
            operation_id: id(if i == 3 {
                DENSE_LINEAR_BF16_OPERATION_ID
            } else {
                operation
            }),
            required_version: VERSION,
            work: ProgramNodeWorkSpec::Fixed,
            inputs: vec![id("value.input"), r.value_id.clone()],
            outputs: vec![id(format!("value.output{i}"))],
            attributes: BTreeMap::from([
                (id("in_features"), SemanticValue::Unsigned(4)),
                (id("out_features"), SemanticValue::Unsigned(1)),
            ]),
        })
        .collect();
    let groups = vec![tensors[..3].iter().map(|t| t.id.clone()).collect()];
    let schema = WeightSchema {
        format_id: id("weight-format.safetensors.dense"),
        layout_id: id("weight-layout.fixture"),
        version: VERSION,
        components,
        tensors,
    };
    let program = ModelProgram::new(
        id("family.fp8-materializer.fixture"),
        vec![id("value.input")],
        vec![ProgramBlock {
            id: "block.main".into(),
            nodes,
        }],
        vec![],
        refs,
        (0..4).map(|i| id(format!("value.output{i}"))).collect(),
    )
    .unwrap();
    (schema, program, groups)
}

#[test]
fn schema_preserves_three_dense_nodes_with_distinct_scales_and_untouched_bf16() {
    let (original, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let prepared = plan::prepare_program(&original, &program, &groups).unwrap();
    let mut scales = BTreeSet::new();
    for id in &groups[0] {
        let tensor = prepared
            .schema
            .tensors
            .iter()
            .find(|t| &t.id == id)
            .unwrap();
        let PhysicalWeightLayout::QuantizedTensor {
            packed_values,
            scales: scale,
            ..
        } = &tensor.physical_layout
        else {
            panic!()
        };
        scales.insert(scale.component_id.clone());
        let expected_sources: Vec<_> = original.components[..3]
            .iter()
            .map(|component| component.id.clone())
            .collect();
        assert_eq!(
            prepared.sources[&packed_values.component_id],
            expected_sources
        );
        assert_eq!(prepared.sources[&scale.component_id], expected_sources);
    }
    assert_eq!(scales.len(), 3);
    assert_eq!(prepared.maximum_group_bytes, 24);
    assert_eq!(prepared.execution_weight_bytes().unwrap(), 32);
    assert_eq!(program.blocks()[0].nodes.len(), 4);
    assert!(prepared
        .schema
        .components
        .iter()
        .any(|c| c == &original.components[3]));
    assert_eq!(
        prepared
            .schema
            .components
            .iter()
            .map(|c| c.physical_bytes().unwrap())
            .sum::<u64>(),
        32
    );
    let mut reversed = groups.clone();
    reversed[0].reverse();
    assert_ne!(
        prepared.schema.fingerprint().unwrap(),
        plan::prepare_program(&original, &program, &reversed)
            .unwrap()
            .schema
            .fingerprint()
            .unwrap()
    );
    let (_, unauthorized, _) = fixture(DENSE_LINEAR_BF16_OPERATION_ID);
    assert!(plan::prepare_program(&original, &unauthorized, &groups).is_err());
    let mut wrong = original.clone();
    wrong.components[0].encoding = WeightEncoding::Dense {
        element_type: ElementType::F16,
    };
    assert!(plan::prepare_program(&wrong, &program, &groups).is_err());
}

#[test]
fn source_component_aliases_are_rejected_before_materialization() {
    let (mut original, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let mut tied = original.tensors[0].clone();
    tied.id = id("weight.q.tied");
    original.tensors.push(tied);
    // The existing source contract forbids two logical trees from sharing a
    // physical leaf. Conversion must not silently legalize such a source.
    assert!(original.validate(program.family_id()).is_err());
    assert!(plan::prepare_program(&original, &program, &groups).is_err());
}

#[test]
fn policy_identity_and_bf16_quality_wire_are_explicit() {
    let (source, program, groups) = fixture(DENSE_LINEAR_FP8_BF16_OPERATION_ID);
    let host = account(32);
    let materializer = Bf16WholeTensorFp8Materializer::new(host.clone(), groups.clone()).unwrap();
    assert!(Bf16WholeTensorFp8Materializer::new(
        host.clone(),
        vec![vec![id("weight.x"), id("weight.x")]]
    )
    .is_err());
    let mut reversed = groups.clone();
    reversed[0].reverse();
    let other = Bf16WholeTensorFp8Materializer::new(host, reversed).unwrap();
    assert_ne!(
        materializer.descriptor().implementation_fingerprint(),
        other.descriptor().implementation_fingerprint()
    );
    let execution = plan::prepare_program(&source, &program, &groups)
        .unwrap()
        .schema;
    let artifact = quality::artifact(materializer.descriptor(), &source, &execution).unwrap();
    let selection = WeightMaterializerSelection::numeric_quality_artifact(
        materializer.descriptor().id().clone(),
        artifact.clone(),
    )
    .unwrap();
    assert!(selection.has_numeric_quality_artifact());
    let json: serde_json::Value = serde_json::from_slice(&artifact).unwrap();
    assert_eq!(
        json["schema_id"],
        NUMERIC_WEIGHT_QUALITY_BF16_ARTIFACT_SCHEMA_ID
    );
    assert!(json["cases"][0].get("actual_f16_bits").is_none());
    for case in json["cases"].as_array().unwrap() {
        let actual = case["actual_bf16_bits"].as_array().unwrap();
        let reference = case["reference_f32_bits"].as_array().unwrap();
        let mut error = 0.0_f64;
        let mut norm = 0.0_f64;
        for (a, r) in actual.iter().zip(reference) {
            let a = f64::from(bf16::from_bits(a.as_u64().unwrap() as u16).to_f32());
            let r = f64::from(f32::from_bits(r.as_u64().unwrap() as u32));
            error += (a - r) * (a - r);
            norm += r * r;
        }
        assert!(error.sqrt() / norm.sqrt().max(1e-6) <= 1.0 / 16.0);
    }
}
