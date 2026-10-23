use super::*;

struct Fixture {
    directory: tempfile::TempDir,
    config: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config.json");
        std::fs::write(
            &config,
            serde_json::to_vec(&json!({
                "_class_name":"QwenImage21Transformer2DModel", "num_layers":1,
                "num_attention_heads":2, "attention_head_dim":8, "context_in_dim":8,
                "in_channels":3, "out_channels":4, "patch_size":1, "mlp_ratio":3,
                "axes_dims_rope":[2,2,4], "causal_condition":true, "eps":0.75
            }))
            .unwrap(),
        )
        .unwrap();
        let shapes: [Vec<usize>; 9] = std::array::from_fn(|i| match i {
            2 | 3 => vec![8],
            6 | 7 => vec![48, 16],
            8 => vec![16, 48],
            _ => vec![16, 16],
        });
        let weights: [Vec<u8>; 9] = std::array::from_fn(|i| {
            (0..shapes[i].iter().product())
                .flat_map(|offset| {
                    let value = if i == 2 || i == 3 {
                        1.0
                    } else {
                        let columns = shapes[i][1];
                        let row = offset / columns;
                        let column = offset % columns;
                        if row % 16 == column % 16 {
                            1.0
                        } else if (row + 1) % 16 == column % 16 {
                            0.125
                        } else {
                            0.0
                        }
                    };
                    half::bf16::from_f32(value).to_bits().to_le_bytes()
                })
                .collect()
        });
        let views = weights
            .iter()
            .enumerate()
            .map(|(i, bytes)| {
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shapes[i].clone(),
                    bytes,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        safetensors::tensor::serialize_to_file(
            Family::WEIGHT_NAMES.into_iter().zip(views),
            &None,
            &directory.path().join("weights.safetensors"),
        )
        .unwrap();
        let weight_map = Family::WEIGHT_NAMES
            .into_iter()
            .map(|name| (name, "weights.safetensors"))
            .collect::<BTreeMap<_, _>>();
        std::fs::write(
            directory.path().join("test.index.json"),
            serde_json::to_vec(&json!({"weight_map":weight_map})).unwrap(),
        )
        .unwrap();
        Self { directory, config }
    }
    fn args(&self, name: &str, reference: Option<&str>) -> Arguments {
        Arguments {
            config: self.config.clone(),
            model_dir: self.directory.path().into(),
            index_file: "test.index.json".into(),
            artifact_dir: self.directory.path().join(name),
            reference_artifact_dir: reference.map(|name| self.directory.path().join(name)),
            reference_memory_bytes: 1 << 20,
            backend: backend::Backend::Reference,
        }
    }
    fn json(&self, name: &str) -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(self.directory.path().join(name)).unwrap()).unwrap()
    }
}

#[test]
fn full_block_executes_nonzero_conditioning_and_checks_all_boundaries_and_retirement() {
    let fixture = Fixture::new();
    run(fixture.args("reference", None)).unwrap();
    let result = fixture.json("reference/result.json");
    assert_eq!(result["validation"]["mode"], "execution_smoke_only");
    assert_eq!(
        result["validation"]["numerical_verification"],
        "not performed"
    );
    assert_eq!(result["validation"]["nonfinite_count"], 0);
    assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
    assert_eq!(result["cleanup"]["released_static_resources"], 1);
    let execution = fixture.json("reference/execution.json");
    assert_eq!(execution["nodes"].as_array().unwrap().len(), 21);
    assert_eq!(execution["initialized_weight_bytes"], 6688);
    assert_eq!(execution["managed_input_bytes"], 384);
    assert_eq!(execution["managed_output_bytes"], 1728);
    let manifest = fixture.json("reference/manifest.json");
    assert_eq!(manifest["outputs"].as_array().unwrap().len(), 21);
    let input = std::fs::read(fixture.directory.path().join("reference/input-00.bf16")).unwrap();
    let output = std::fs::read(fixture.directory.path().join("reference/output-20.bf16")).unwrap();
    assert_ne!(
        input, output,
        "nonzero gates must exercise the residual branches"
    );
    // This checks differential plumbing, not an independent mathematical oracle.
    run(fixture.args("candidate", Some("reference"))).unwrap();
    let result = fixture.json("candidate/result.json");
    assert_eq!(result["validation"]["mode"], "reference_differential");
    assert_eq!(result["validation"]["failure_count"], 0);
    for port in result["validation"]["ports"].as_array().unwrap() {
        assert_eq!(port["elements"], port["compared_elements"]);
        assert_eq!(port["maximum_absolute_error"], 0.0);
    }
}

#[test]
fn reference_identity_and_output_tampering_fail_before_candidate_execution() {
    let fixture = Fixture::new();
    run(fixture.args("reference", None)).unwrap();
    let original = std::fs::read(&fixture.config).unwrap();
    let mut different = original.clone();
    different.push(b'\n');
    std::fs::write(&fixture.config, different).unwrap();
    assert!(run(fixture.args("wrong-config", Some("reference")))
        .unwrap_err()
        .to_string()
        .contains("identity mismatch"));
    assert!(!fixture.directory.path().join("wrong-config").exists());
    std::fs::write(&fixture.config, original).unwrap();
    let output = fixture.directory.path().join("reference/output-20.bf16");
    let mut bytes = std::fs::read(&output).unwrap();
    bytes[0] ^= 1;
    std::fs::write(output, bytes).unwrap();
    assert!(run(fixture.args("wrong-output", Some("reference")))
        .unwrap_err()
        .to_string()
        .contains("digest"));
    assert!(!fixture.directory.path().join("wrong-output").exists());
}

#[test]
fn differential_uses_declared_error_bound_and_rejects_nonfinite_values() {
    let spec = probe::OutputSpec {
        value_id: "value.test".into(),
        file: "output.bf16".into(),
        shape: vec![1, 1],
        byte_len: 2,
    };
    let bytes = |value: f32| half::bf16::from_f32(value).to_bits().to_le_bytes();
    let reference = bytes(1.0);
    let compare = |value| {
        serde_json::to_value(evidence::compare(&spec, &bytes(value), Some(&reference)).unwrap())
            .unwrap()
    };
    assert_eq!(compare(1.03125)["failure_count"], 0);
    assert_eq!(compare(1.125)["failure_count"], 1);
    assert_eq!(compare(f32::NAN)["nonfinite_count"], 1);
    assert_eq!(compare(f32::INFINITY)["failure_count"], 1);
    assert!(evidence::compare(&spec, &[], Some(&reference)).is_err());
}

#[test]
fn cli_rejects_unknown_duplicate_and_unavailable_backend_options() {
    let base = ["--config", "c", "--model-dir", "m", "--artifact-dir", "a"];
    let parse = |tail: &[&str]| {
        Arguments::parse(
            base.into_iter()
                .chain(tail.iter().copied())
                .map(OsString::from),
        )
    };
    assert!(parse(&[])
        .unwrap()
        .unwrap()
        .reference_artifact_dir
        .is_none());
    assert!(parse(&["--reference-artifact-dir", "r"])
        .unwrap()
        .unwrap()
        .reference_artifact_dir
        .is_some());
    assert!(parse(&["--unknown", "1"]).is_err());
    assert!(parse(&["--artifact-dir", "duplicate"]).is_err());
    assert!(parse(&["--backend", "cuda"]).is_err());
    assert_eq!(
        parse(&[]).unwrap().unwrap().reference_memory_bytes,
        DEFAULT_REFERENCE_MEMORY_BYTES
    );
    assert_eq!(
        parse(&["--reference-memory-bytes", "1048576"])
            .unwrap()
            .unwrap()
            .reference_memory_bytes,
        1 << 20
    );
    for value in ["0", "-1", "18446744073709551615"] {
        assert!(parse(&["--reference-memory-bytes", value]).is_err());
    }
}

#[test]
fn reference_budget_rejects_weights_that_exceed_declared_capacity() {
    let fixture = Fixture::new();
    let mut args = fixture.args("undersized", None);
    args.reference_memory_bytes = 64;
    assert!(run(args).is_err());
    assert!(!fixture
        .directory
        .path()
        .join("undersized/result.json")
        .exists());
}
