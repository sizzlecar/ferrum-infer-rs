use serde_json::{json, Value};

use super::*;

fn real_config() -> Value {
    json!({
        "_class_name": "QwenImage21Transformer2DModel", "_diffusers_version": "0.41.0.dev0",
        "num_layers": 32, "num_attention_heads": 32, "attention_head_dim": 128,
        "context_in_dim": 4096, "in_channels": 64, "out_channels": 64, "patch_size": 1,
        "mlp_ratio": 3, "axes_dims_rope": [16,56,56], "causal_condition": true, "eps": 0.000001
    })
}

fn tiny_config() -> Value {
    let mut config = real_config();
    for (name, value) in [
        ("num_layers", 1),
        ("num_attention_heads", 2),
        ("attention_head_dim", 8),
        ("context_in_dim", 12),
        ("in_channels", 4),
        ("out_channels", 4),
    ] {
        config[name] = json!(value);
    }
    config["axes_dims_rope"] = json!([2, 2, 4]);
    config
}

fn config_from(value: &Value) -> Result<QwenImage21Config> {
    QwenImage21Config::from_hf_json(
        &serde_json::to_vec(value).unwrap(),
        StructureLimits::default(),
    )
}

struct ShardFixture {
    name: String,
    header: Value,
    payload_bytes: u64,
}
struct Fixture {
    config: Value,
    index: Value,
    shards: Vec<ShardFixture>,
}

impl Fixture {
    fn new() -> Self {
        let config = tiny_config();
        let expected = ExpectedWeights::from_config(
            &config_from(&config).unwrap(),
            StructureLimits::default(),
        )
        .unwrap();
        let mut shards: Vec<_> = ["first.safetensors", "second.safetensors"]
            .into_iter()
            .map(|name| ShardFixture {
                name: name.to_owned(),
                header: json!({"__metadata__":{"format":"pt"}}),
                payload_bytes: 0,
            })
            .collect();
        let mut weight_map = serde_json::Map::new();
        for (index, tensor) in expected.tensors().values().enumerate() {
            let shard = &mut shards[index % 2];
            let start = shard.payload_bytes;
            shard.payload_bytes += tensor.bytes();
            shard.header[tensor.external_name()] = json!({"dtype":"BF16", "shape":tensor.dimensions(), "data_offsets":[start,shard.payload_bytes]});
            weight_map.insert(tensor.external_name().to_owned(), json!(shard.name));
        }
        let index =
            json!({"metadata":{"total_size":expected.payload_bytes()},"weight_map":weight_map});
        Self {
            config,
            index,
            shards,
        }
    }

    fn header_bytes(&self) -> Vec<Vec<u8>> {
        self.shards
            .iter()
            .map(|shard| serde_json::to_vec(&shard.header).unwrap())
            .collect()
    }
    fn inspect(&self) -> Result<QwenImage21Structure> {
        self.inspect_raw(
            &serde_json::to_vec(&self.index).unwrap(),
            &self.header_bytes(),
            StructureLimits::default(),
        )
    }
    fn inspect_raw(
        &self,
        index: &[u8],
        headers: &[Vec<u8>],
        limits: StructureLimits,
    ) -> Result<QwenImage21Structure> {
        let evidence = self
            .shards
            .iter()
            .zip(headers)
            .map(|(shard, header)| {
                HeaderEvidence::new(
                    &shard.name,
                    header,
                    8 + header.len() as u64 + shard.payload_bytes,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        QwenImage21Structure::inspect(
            &serde_json::to_vec(&self.config).unwrap(),
            index,
            &evidence,
            limits,
        )
    }
    fn first_tensor(&self) -> String {
        self.shards[0]
            .header
            .as_object()
            .unwrap()
            .keys()
            .find(|name| *name != "__metadata__")
            .unwrap()
            .clone()
    }
}

#[test]
fn real_configuration_generates_exact_single_stream_bf16_structure() {
    let config = config_from(&real_config()).unwrap();
    assert_eq!(config.hidden_size(), 4096);
    assert_eq!(config.mlp_hidden_size(), 12288);
    assert_eq!(config.rope_axes(), &[16, 56, 56]);
    let expected = ExpectedWeights::from_config(&config, StructureLimits::default()).unwrap();
    assert_eq!(expected.tensors().len(), 297);
    assert_eq!(expected.parameter_count(), 7_115_124_736);
    assert_eq!(expected.payload_bytes(), 14_230_249_472);
    for (name, shape) in [
        ("img_in.weight", vec![4096, 64]),
        ("modulation.1.weight", vec![16384, 4096]),
        (
            "time_text_embed.timestep_embedder.linear_1.weight",
            vec![4096, 256],
        ),
        ("txt_in.text_norm.weight", vec![4096]),
        ("norm_out.linear.weight", vec![4096, 4096]),
        ("proj_out.weight", vec![64, 4096]),
        ("transformer_blocks.31.attn.norm_q.weight", vec![128]),
        (
            "transformer_blocks.0.attn.to_out.0.weight",
            vec![4096, 4096],
        ),
        (
            "transformer_blocks.31.img_mlp.gate_layer.weight",
            vec![12288, 4096],
        ),
        (
            "transformer_blocks.31.img_mlp.out.weight",
            vec![4096, 12288],
        ),
    ] {
        assert_eq!(expected.tensors()[name].dimensions(), shape);
    }
    assert!(!expected
        .tensors()
        .keys()
        .any(|name| name.ends_with("bias") || name.contains("txt_mlp")));
    let schema = expected.weight_schema().unwrap();
    assert_eq!(schema.components.len(), 297);
    assert!(schema
        .components
        .iter()
        .all(|component| component.physical_element_type() == ferrum_runtime::ElementType::Bf16));
    assert_eq!(
        schema
            .components
            .iter()
            .map(|component| component.physical_bytes().unwrap())
            .sum::<u64>(),
        expected.payload_bytes()
    );
}

#[test]
fn config_rejects_inconsistent_dimensions_unsupported_patch_and_overflow() {
    for (field, value) in [
        ("_class_name", json!("QwenImageTransformer2DModel")),
        ("num_layers", json!(0)),
        ("num_attention_heads", json!(0)),
        ("patch_size", json!(2)),
        ("eps", json!(0)),
        ("eps", json!(-0.001)),
        ("axes_dims_rope", json!([3, 1, 4])),
        ("axes_dims_rope", json!([2, 2, 2])),
        ("axes_dims_rope", json!([u64::MAX - 1, 2, 2])),
        ("num_attention_heads", json!(u64::MAX)),
        ("mlp_ratio", json!(u64::MAX)),
        ("num_layers", json!(u64::MAX)),
        ("context_in_dim", json!(u64::MAX)),
        ("unknown_semantic_option", json!(true)),
    ] {
        let mut config = tiny_config();
        config[field] = value;
        assert!(config_from(&config).is_err(), "accepted invalid {field}");
    }
    let mut duplicate = serde_json::to_string(&tiny_config()).unwrap();
    duplicate.insert_str(1, "\"num_layers\":1,");
    assert!(
        QwenImage21Config::from_hf_json(duplicate.as_bytes(), StructureLimits::default()).is_err()
    );
}

#[test]
fn canonical_config_deserialization_preserves_validation() {
    let config = config_from(&tiny_config()).unwrap();
    let mut canonical = serde_json::to_value(&config).unwrap();
    assert_eq!(
        serde_json::from_value::<QwenImage21Config>(canonical.clone()).unwrap(),
        config
    );
    canonical["heads"] = json!(0);
    assert!(serde_json::from_value::<QwenImage21Config>(canonical).is_err());
}

#[test]
fn detached_headers_prove_structure_only_and_order_does_not_change_identity() {
    let mut fixture = Fixture::new();
    let first = fixture.inspect().unwrap();
    assert_eq!(first.tensors().len(), 18);
    assert_eq!(
        first.payload_verification(),
        PayloadVerification::NotVerified
    );
    assert_eq!(first.weight_schema().unwrap().tensors.len(), 18);
    fixture.shards.reverse();
    assert_eq!(fixture.inspect().unwrap(), first);
    fixture.shards[0].header["__metadata__"]["note"] = json!("changed metadata only");
    let changed = fixture.inspect().unwrap();
    assert_eq!(changed.payload_bytes(), first.payload_bytes());
    assert_ne!(
        changed.structure_fingerprint(),
        first.structure_fingerprint()
    );
    assert_eq!(
        first.payload_verification(),
        PayloadVerification::NotVerified
    );
}

#[test]
fn index_must_exactly_identify_each_tensor_and_its_shard() {
    let mut fixture = Fixture::new();
    let name = fixture.first_tensor();
    fixture.index["weight_map"][&name] = json!(fixture.shards[1].name);
    assert!(fixture
        .inspect()
        .unwrap_err()
        .to_string()
        .contains("index-declared shard"));
    let mut fixture = Fixture::new();
    fixture.index["metadata"]["total_size"] = json!(1);
    assert!(fixture.inspect().is_err());
    let mut fixture = Fixture::new();
    let name = fixture.first_tensor();
    let old = fixture.index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(&name)
        .unwrap();
    fixture.index["weight_map"]["unexpected.weight"] = old;
    assert!(fixture.inspect().is_err());
    let mut fixture = Fixture::new();
    fixture.index["weight_map"]
        .as_object_mut()
        .unwrap()
        .remove(&name);
    assert!(fixture.inspect().is_err());
}

#[test]
fn duplicate_json_keys_are_rejected_before_map_overwrite() {
    let fixture = Fixture::new();
    let name = fixture.first_tensor();
    let value = serde_json::to_string(&fixture.shards[0].header[&name]).unwrap();
    let mut headers = fixture.header_bytes();
    let mut header = String::from_utf8(headers[0].clone()).unwrap();
    header.insert_str(1, &format!("{name:?}:{value},"));
    headers[0] = header.into_bytes();
    let index = serde_json::to_vec(&fixture.index).unwrap();
    assert!(fixture
        .inspect_raw(&index, &headers, StructureLimits::default())
        .unwrap_err()
        .to_string()
        .contains("duplicate header key"));
    let index = serde_json::to_string(&fixture.index).unwrap().replacen(
        "\"weight_map\":{",
        &format!("\"weight_map\":{{{name:?}:{:?},", fixture.shards[0].name),
        1,
    );
    assert!(fixture
        .inspect_raw(
            index.as_bytes(),
            &fixture.header_bytes(),
            StructureLimits::default()
        )
        .unwrap_err()
        .to_string()
        .contains("duplicate metadata key"));
}

#[test]
fn headers_reject_shape_dtype_missing_and_unknown_tensors() {
    for (field, value) in [
        ("shape", json!([u64::MAX, 2])),
        ("shape", json!([0])),
        ("shape", json!([])),
        ("shape", json!([1])),
        ("dtype", json!("F16")),
        ("data_offsets", json!([1, 0])),
    ] {
        let mut fixture = Fixture::new();
        let name = fixture.first_tensor();
        fixture.shards[0].header[&name][field] = value;
        assert!(fixture.inspect().is_err(), "accepted invalid {field}");
    }
    let mut fixture = Fixture::new();
    let name = fixture.first_tensor();
    fixture.shards[0]
        .header
        .as_object_mut()
        .unwrap()
        .remove(&name);
    assert!(fixture.inspect().is_err());
    let mut fixture = Fixture::new();
    let value = fixture.shards[0]
        .header
        .as_object_mut()
        .unwrap()
        .remove(&name)
        .unwrap();
    fixture.shards[0].header["unknown.weight"] = value;
    assert!(fixture.inspect().is_err());
}

#[test]
fn header_intervals_must_exactly_partition_the_declared_payload() {
    for displacement in [-2_i64, 2] {
        let mut fixture = Fixture::new();
        let name = fixture.shards[0]
            .header
            .as_object()
            .unwrap()
            .iter()
            .find(|(name, tensor)| {
                *name != "__metadata__" && tensor["data_offsets"][0].as_u64().unwrap() > 2
            })
            .unwrap()
            .0
            .clone();
        let tensor = &mut fixture.shards[0].header[&name];
        let start = tensor["data_offsets"][0].as_u64().unwrap() as i64 + displacement;
        let end = tensor["data_offsets"][1].as_u64().unwrap() as i64 + displacement;
        tensor["data_offsets"] = json!([start, end]);
        assert!(fixture.inspect().is_err());
    }
    let mut fixture = Fixture::new();
    fixture.shards[0].payload_bytes += 2;
    assert!(fixture
        .inspect()
        .unwrap_err()
        .to_string()
        .contains("declared file length"));
    let mut fixture = Fixture::new();
    fixture.shards[0].payload_bytes -= 2;
    assert!(fixture.inspect().is_err());
}

#[test]
fn shard_set_and_paths_are_explicit_and_bounded() {
    let fixture = Fixture::new();
    let config = serde_json::to_vec(&fixture.config).unwrap();
    let index = serde_json::to_vec(&fixture.index).unwrap();
    let headers = fixture.header_bytes();
    let one = HeaderEvidence::new(
        &fixture.shards[0].name,
        &headers[0],
        headers[0].len() as u64 + 8 + fixture.shards[0].payload_bytes,
    )
    .unwrap();
    assert!(
        QwenImage21Structure::inspect(&config, &index, &[one], StructureLimits::default()).is_err()
    );
    assert!(QwenImage21Structure::inspect(
        &config,
        &index,
        &[one, one],
        StructureLimits::default()
    )
    .unwrap_err()
    .to_string()
    .contains("duplicate header"));
    for path in [
        "../x.safetensors",
        "/x.safetensors",
        "a/../x.safetensors",
        "a\\x.safetensors",
        "C:/x.safetensors",
        "a//x.safetensors",
        "x.bin",
    ] {
        assert!(HeaderEvidence::new(path, b"{}", 12).is_err());
    }
    assert!(HeaderEvidence::new("x.safetensors", b"{}", 10).is_err());
    assert!(HeaderEvidence::new("x.safetensors", b"", 10).is_err());
}

#[test]
fn structure_limits_fail_before_unbounded_inventory_or_parsing() {
    let fixture = Fixture::new();
    let headers = fixture.header_bytes();
    let index = serde_json::to_vec(&fixture.index).unwrap();
    for limits in [
        StructureLimits::new(1, 100000, 100, 2).unwrap(),
        StructureLimits::new(100000, 1, 100, 2).unwrap(),
        StructureLimits::new(100000, 100000, 17, 2).unwrap(),
        StructureLimits::new(100000, 100000, 100, 1).unwrap(),
    ] {
        assert!(fixture.inspect_raw(&index, &headers, limits).is_err());
    }
    assert!(StructureLimits::new(0, 1, 1, 1).is_err());
    assert!(StructureLimits::new(isize::MAX as u64, 1, 1, 1).is_err());
    assert!(StructureLimits::new(1, isize::MAX as u64 + 1, 1, 1).is_err());
    assert!(StructureLimits::new(1, 1, 0, 1).is_err());
    assert!(StructureLimits::new(1, 1, 1, 0).is_err());
}

#[test]
fn header_padding_is_part_of_the_file_offset_evidence() {
    let fixture = Fixture::new();
    let config = serde_json::to_vec(&fixture.config).unwrap();
    let index = serde_json::to_vec(&fixture.index).unwrap();
    let mut headers = fixture.header_bytes();
    headers[0].extend_from_slice(b"   ");
    fixture
        .inspect_raw(&index, &headers, StructureLimits::default())
        .unwrap();
    let full_first_length = headers[0].len() as u64 + 8 + fixture.shards[0].payload_bytes;
    let unpadded_len = headers[0].len() - 3;
    headers[0].truncate(unpadded_len);
    let evidence = [
        HeaderEvidence::new(&fixture.shards[0].name, &headers[0], full_first_length).unwrap(),
        HeaderEvidence::new(
            &fixture.shards[1].name,
            &headers[1],
            headers[1].len() as u64 + 8 + fixture.shards[1].payload_bytes,
        )
        .unwrap(),
    ];
    assert!(
        QwenImage21Structure::inspect(&config, &index, &evidence, StructureLimits::default())
            .unwrap_err()
            .to_string()
            .contains("declared file length")
    );
}
