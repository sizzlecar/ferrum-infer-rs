use super::*;
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use serde_json::json;
#[path = "../../../examples/support/tensor_component.rs"]
mod shared;

fn id<T: TryFrom<String, Error = VNextError>>(s: impl Into<String>) -> T {
    T::try_from(s.into()).unwrap()
}
fn bytes(values: impl IntoIterator<Item = u16>) -> Vec<u8> {
    values.into_iter().flat_map(u16::to_le_bytes).collect()
}
fn b(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}
fn decode(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

#[test]
fn real_metadata_produces_exact_prefix_schema_and_no_unexecuted_weights() {
    let raw = json!({"_class_name":"AutoencoderKLQwenImage21","z_dim":64,"base_dim":96,"decoder_base_dim":144,
        "dim_mult":[1,2,4,8,8],"patch_size":null,"dropout":0.0});
    let c = QwenImage21VaeStemConfig::from_hf_json(&serde_json::to_vec(&raw).unwrap(), 1, 4, 6)
        .unwrap();
    assert_eq!(c.hidden_channels(), 1152);
    assert_eq!(c.latent_channels(), 64);
    let family = QwenImage21VaeStemFamily::new().unwrap();
    let stored = family.weights(&c).unwrap();
    let schema = stored.stored_schema().unwrap();
    assert_eq!(schema.tensors.len(), 10);
    assert_eq!(schema.components.len(), 10);
    assert_eq!(
        schema
            .components
            .iter()
            .map(|w| w.physical_bytes().unwrap())
            .sum::<u64>(),
        49_122_688
    );
    assert!(schema.components.iter().all(|w| w.required));
    let weights = c.expected_weights().unwrap();
    for (index, shape) in [
        (0, vec![64, 64, 1, 1]),
        (2, vec![1152, 64, 3, 3]),
        (4, vec![1152, 1, 1, 1]),
        (5, vec![1152, 1152, 3, 3]),
        (7, vec![1152, 1, 1, 1]),
    ] {
        assert_eq!(weights[index].dimensions(), shape);
    }
    let p = family.numerical_profiles(&c).unwrap();
    let graph = family.semantic_program(&c, &p.profiles()[0]).unwrap();
    assert_eq!(graph.inputs(), [id(QwenImage21VaeStemFamily::INPUT_ID)]);
    assert_eq!(graph.outputs().len(), 9);
    assert_eq!(graph.blocks()[0].nodes.len(), 9);
    assert_eq!(
        graph.blocks()[0].nodes[8].inputs,
        [
            id(QwenImage21VaeStemFamily::OUTPUT_IDS[7]),
            id(QwenImage21VaeStemFamily::OUTPUT_IDS[1])
        ]
    );
    assert_eq!(graph.input_tensor_specs(), &c.input_tensor_specs().unwrap());
    for node in &graph.blocks()[0].nodes {
        if node.operation_id.as_str() == CONV2D_BF16_OPERATION_ID {
            assert_eq!(
                node.attributes[&id("padding_mode")],
                SemanticValue::Text("explicit_zero".into())
            );
        }
    }
    assert_eq!(c.compile_options().unwrap().tensor_specs().len(), 10);
}

#[test]
fn checked_nchw_geometry_and_wire_reject_invalid_scope_or_overflow() {
    for dimensions in [
        (0, 1, 1, 2, 3),
        (1, 0, 1, 2, 3),
        (1, 1, 0, 2, 3),
        (1, 1, 1, 0, 3),
        (1, 1, 1, 2, 0),
        (u64::MAX, 1, 1, 2, 3),
        (1, 1, 1, u64::MAX, 3),
    ] {
        assert!(QwenImage21VaeStemConfig::new(
            dimensions.0,
            dimensions.1,
            dimensions.2,
            dimensions.3,
            dimensions.4
        )
        .is_err());
    }
    let c = QwenImage21VaeStemConfig::new(1, 1, 2, 2, 3).unwrap();
    let mut wire = serde_json::to_value(c).unwrap();
    wire["frames"] = json!(2);
    assert!(serde_json::from_value::<QwenImage21VaeStemConfig>(wire).is_err());
    for field in ["_class_name", "patch_size", "dropout"] {
        let mut raw = json!({"_class_name":"AutoencoderKLQwenImage21","z_dim":1,"decoder_base_dim":1,"dim_mult":[1],"patch_size":null,"dropout":0.0});
        raw[field] = match field {
            "_class_name" => json!("other"),
            "patch_size" => json!(2),
            _ => json!(0.1),
        };
        assert!(QwenImage21VaeStemConfig::from_hf_json(
            &serde_json::to_vec(&raw).unwrap(),
            1,
            2,
            3
        )
        .is_err());
    }
}

#[test]
fn nine_node_managed_prefix_matches_distinct_hand_computed_boundaries_and_retires() {
    let c = QwenImage21VaeStemConfig::new(1, 1, 2, 2, 3).unwrap();
    let f = QwenImage21VaeStemFamily::new().unwrap();
    let family = TypedFamilyRegistration::new(f)
        .prepare_with_profile(
            &serde_json::to_value(&c).unwrap(),
            &NumericalProfileId::new(QwenImage21VaeStemFamily::REFERENCE_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let expected_weights = c.expected_weights().unwrap();
    let mut payloads = BTreeMap::new();
    for (index, w) in expected_weights.iter().enumerate() {
        let mut data = vec![0_u16; (w.bytes() / 2) as usize];
        match index {
            0 => data[0] = b(2.0),
            1 | 3 | 9 => data[0] = b(0.5),
            2 => data[4] = b(1.0), // conv_in center only
            4 | 7 => data[0] = b(1.0),
            5 => data[4] = b(2.0),
            6 => data[0] = b(0.125),
            8 => data[5] = b(1.0), // cross-correlation: immediate right neighbor
            _ => unreachable!(),
        }
        payloads.insert(
            w.external_name().to_string(),
            (
                w.dimensions()
                    .iter()
                    .map(|&n| n as usize)
                    .collect::<Vec<_>>(),
                bytes(data),
            ),
        );
    }
    let directory = tempfile::tempdir().unwrap();
    let views = payloads
        .iter()
        .map(|(name, (shape, data))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, shape.clone(), data)
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    let source = [-2., -1., 1., 2., 3., -3., 2., 1., -1., -2., -3., 3.];
    let input = bytes(source.map(b));
    // With C=1, both L2 normalizations are exactly sign(x). The two
    // nonzero SiLU values are independently established BF16 constants.
    // The final conv samples the right neighbor, and explicit padding supplies
    // zero at each row edge. Bias/residual ties are rounded to even by hand.
    let signs = [-1., -1., 1., 1., 1., -1., 1., 1., -1., -1., -1., 1.];
    let silu = signs.map(|s| if s > 0.0 { 0x3f3b } else { 0xbe8a });
    let conv1 = signs.map(|s| if s > 0.0 { 0x3fcb } else { 0xbed4 });
    let conv2 = [
        0x3e6c, 0x3f9e, 0x3f00, 0x3f9e, 0x3e6c, 0x3f00, 0x3f9e, 0x3e6c, 0x3f00, 0x3e6c, 0x3f9e,
        0x3f00,
    ];
    let residual = [
        0xc031, 0x3e70, 0x4060, 0x40c8, 0x40e7, 0xc090, 0x40c8, 0x404f, 0xbf00, 0xc031, 0xc071,
        0x40f0,
    ];
    let expected = [
        source.map(|x| b(2.0 * x + 0.5)).to_vec(),
        source.map(|x| b(2.0 * x + 1.0)).to_vec(),
        signs.map(b).to_vec(),
        silu.to_vec(),
        conv1.to_vec(),
        signs.map(b).to_vec(),
        silu.to_vec(),
        conv2.to_vec(),
        residual.to_vec(),
    ];
    let composition =
        ReferenceVNextComposition::create(id("device.reference.vae_stem.test")).unwrap();
    let capacity = composition.runtime().descriptor().total_memory_bytes;
    let (_, cleanup) = shared::execute_stored_component_with_options(
        &family,
        &archive,
        shared::StoredComponentRuntime {
            runtime: composition.runtime().clone(),
            registry: composition.registry(),
            catalog: composition.catalog(),
            materializers: composition.weight_materializers(),
            capacity_bytes: capacity,
            staging_bytes: 128,
            provenance: json!({"scope":"hand-computed single-channel prefix composition fixture"}),
        },
        vec![shared::ComponentInput {
            value_id: id(QwenImage21VaeStemFamily::INPUT_ID),
            tensor: c.tensor(1).unwrap(),
            bytes: &input,
        }],
        c.compile_options().unwrap(),
        |outputs, _| {
            assert_eq!(outputs.len(), 9);
            for (id, expected) in QwenImage21VaeStemFamily::OUTPUT_IDS
                .into_iter()
                .zip(expected)
            {
                let actual = outputs
                    .iter()
                    .find(|o| o.value_id().as_str() == id)
                    .unwrap();
                assert_eq!(actual.tensor().dimensions(), [2, 1, 2, 3]);
                assert_eq!(decode(actual.bytes().bytes()), expected, "{id}");
            }
            Ok(json!({"matched":true}))
        },
    )
    .unwrap();
    assert_eq!(cleanup["managed_host_committed_bytes"], 0);
    assert_eq!(cleanup["managed_host_allocated_bytes"], 0);
    assert_eq!(cleanup["released_static_resources"], 1);
}
