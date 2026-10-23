use super::*;

fn geometry() -> Geometry {
    Geometry {
        rows: 1,
        heads: 1,
        head_dim: 2,
        mlp_width: 4,
        epsilon_f32_bits: 1e-6_f32.to_bits(),
        scale_f32_bits: 0.5_f32.to_bits(),
    }
}
fn case() -> Case {
    let geometry = geometry();
    let segment = |name: String, bytes| Segment {
        location: SegmentLocation {
            path: PathBuf::from(format!("/original/{name}")),
            offset: 0,
            bytes,
        },
        sha256: "0".repeat(64),
    };
    let inputs = geometry
        .input_shapes()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, shape)| {
            segment(
                format!("input-{i}"),
                tensor_bytes(&shape, if i == 5 { 4 } else { 2 }).unwrap(),
            )
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let weights = geometry
        .weight_shapes()
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, shape)| segment(format!("weight-{i}"), tensor_bytes(&shape, 2).unwrap()))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    Case {
        geometry,
        policy: MathPolicy::LibraryDefaults,
        inputs,
        weights,
    }
}
fn config() -> native::Config {
    serde_json::from_value(json!({
        "torch_root":"/explicit/torch", "library_directories":["/explicit/torch/lib"],
        "executable":{"mode":"prebuilt","path":"/explicit/oracle","sha256":"0".repeat(64)},
        "cxx11_abi":1,"expected_torch_version":"test-version","expected_cuda_runtime_soname":"libcudart.so.13",
        "deployment_profile":"test-only","device":0,"timeout_seconds":30,"maximum_payload_bytes":1024
    })).unwrap()
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn metadata(case: &Case) -> serde_json::Value {
    json!({"operation":"aten.qwen_image21.block0", "arithmetic":ARITHMETIC,
        "torch_version":"test-version","torch_config":"fixture","cxx11_abi":1,"device":0,
        "policy":case.policy,"geometry":case.geometry,"sources":case.sources().map(|s| &s.location).collect::<Vec<_>>(),
        "boundaries":case.geometry.boundaries().unwrap(),"bf16_reduction_option":0,"allow_tf32":false,
        "float32_matmul_precision":0,"blas_preferred_backend":0,"deterministic_algorithms":false,
        "cuda_runtime_version":13000,"loaded_libraries":[]})
}
fn frame(metadata: &serde_json::Value, bytes: u64) -> Vec<u8> {
    let encoded = serde_json::to_vec(metadata).unwrap();
    let mut frame = b"FBLKOR01".to_vec();
    frame.extend(1_u32.to_le_bytes());
    frame.extend((encoded.len() as u32).to_le_bytes());
    frame.extend(encoded);
    frame.extend(bytes.to_le_bytes());
    frame.resize(frame.len() + bytes as usize, 0);
    frame
}

#[test]
fn full_matrix_budget_accepts_mlp_weights_but_rejects_individual_and_aggregate_overflow() {
    let g = Geometry {
        rows: 2,
        heads: 32,
        head_dim: 128,
        mlp_width: 12288,
        ..geometry()
    };
    g.validate().unwrap();
    assert_eq!(
        tensor_bytes(&g.weight_shapes().unwrap()[6], 2).unwrap(),
        96 * 1024 * 1024
    );
    assert_eq!(g.output_bytes().unwrap(), 432 * 1024);
    assert_eq!(
        super::super::linear::MAX_WEIGHT_BYTES,
        64 * 1024 * 1024,
        "existing linear oracle contract is unchanged"
    );
    let maximum_total = Geometry {
        mlp_width: 16383,
        ..g
    };
    maximum_total.validate().unwrap();
    // Three individually permitted 128 MiB matrices plus four 32 MiB matrices
    // and the two gamma vectors exceed the total budget by 512 bytes.
    assert!(Geometry {
        mlp_width: 16384,
        ..g
    }
    .validate()
    .is_err());
    assert!(Geometry {
        mlp_width: 16385,
        ..g
    }
    .validate()
    .is_err());
    assert!(Geometry {
        rows: u64::MAX,
        ..g
    }
    .validate()
    .is_err());
    assert!(Geometry {
        epsilon_f32_bits: f32::NAN.to_bits(),
        ..g
    }
    .validate()
    .is_err());
    let mut wrong = case();
    wrong.weights[6].location.bytes -= 2;
    assert!(wrong.validate().is_err());
}

#[test]
fn block_protocol_requires_complete_ordered_sources_boundaries_and_observed_math() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("block.bin");
    let case = case();
    let count = case.geometry.output_bytes().unwrap();
    let original = metadata(&case);
    let valid = frame(&original, count);
    std::fs::write(&path, &valid).unwrap();
    let host = host(count as usize);
    let output = parse(&path, &config(), &case, &host).unwrap();
    assert!(host.owns(&output.bytes));
    assert_eq!(host.usage().unwrap().committed_bytes(), count as usize);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let mut variants = Vec::new();
    let mut changed = original.clone();
    changed["allow_tf32"] = json!(true);
    variants.push(changed);
    let mut changed = original.clone();
    changed["sources"][0]["offset"] = json!(1);
    variants.push(changed);
    let mut changed = original.clone();
    changed["boundaries"].as_array_mut().unwrap().pop();
    variants.push(changed);
    let mut changed = original.clone();
    changed["boundaries"].as_array_mut().unwrap().swap(0, 1);
    variants.push(changed);
    let mut changed = original.clone();
    changed["geometry"]["heads"] = json!(2);
    variants.push(changed);
    let mut changed = original.clone();
    changed["operation"] = json!("aten.linear.no_bias");
    variants.push(changed);
    for changed in variants {
        std::fs::write(&path, frame(&changed, count)).unwrap();
        assert!(parse(&path, &config(), &case, &host).is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
    let mut strict = case.clone();
    strict.policy = MathPolicy::NoReducedPrecision;
    let mut observed = metadata(&strict);
    std::fs::write(&path, frame(&observed, count)).unwrap();
    assert!(parse(&path, &config(), &strict, &host).is_err());
    observed["bf16_reduction_option"] = json!(1);
    std::fs::write(&path, frame(&observed, count)).unwrap();
    drop(parse(&path, &config(), &strict, &host).unwrap());
    for malformed in [
        valid[..valid.len() - 1].to_vec(),
        [valid.as_slice(), &[0]].concat(),
        [b"FLINOR01".as_slice(), &valid[8..]].concat(),
        frame(&original, count - 2),
    ] {
        std::fs::write(&path, malformed).unwrap();
        assert!(parse(&path, &config(), &case, &host).is_err());
    }
    std::fs::write(&path, valid).unwrap();
    let too_small = self::host(count as usize - 1);
    assert!(parse(&path, &config(), &case, &too_small).is_err());
    assert_eq!(too_small.usage().unwrap().committed_bytes(), 0);
    let mut capped = config();
    capped.maximum_payload_bytes = count - 1;
    assert!(parse(&path, &capped, &case, &host).is_err());
}

#[test]
fn original_input_selection_uses_semantic_ids_and_rejects_intermediate_substitution() {
    let dir = tempfile::tempdir().unwrap();
    let g = geometry();
    let mut inputs = Vec::new();
    for (index, shape) in g.input_shapes().unwrap().into_iter().enumerate() {
        let bytes = if index == 5 {
            [1_f32.to_le_bytes(), 0_f32.to_le_bytes()].concat()
        } else {
            vec![index as u8; tensor_bytes(&shape, 2).unwrap() as usize]
        };
        let file = format!("input-{index}.bin");
        std::fs::write(dir.path().join(&file), &bytes).unwrap();
        inputs.push(source::Tensor {
            value_id: Block::INPUT_IDS[index].into(),
            file,
            dtype: if index == 5 { "F32" } else { "BF16" }.into(),
            shape,
            bytes: bytes.len() as u64,
            sha256: native::bytes_sha(&bytes),
        });
    }
    inputs.reverse();
    let mut identity = source::Identity {
        canonical_config: serde_json::Value::Null,
        config_file_sha256: "0".repeat(64),
        inputs,
        weights: vec![],
    };
    let selected = replay::original_inputs(dir.path(), &identity, g).unwrap();
    assert!(selected[0].location.path.ends_with("input-0.bin"));
    assert!(selected[5].location.path.ends_with("input-5.bin"));
    // An otherwise valid saved intermediate cannot fill a missing original X
    // merely because its shape matches. No output-port list is an input source.
    identity.inputs[5].value_id = Block::MODULATED1_ID.into();
    assert!(replay::original_inputs(dir.path(), &identity, g).is_err());
    identity.inputs[5].value_id = Block::INPUT_ID.into();
    identity.inputs[5].file = "../outside.bin".into();
    assert!(replay::original_inputs(dir.path(), &identity, g).is_err());
    std::fs::write(&selected[0].location.path, [1, 0, 0, 0]).unwrap();
    assert!(
        selected[0].verify().is_err(),
        "after-child mutation must invalidate the original digest"
    );
    let mut malformed = selected[0].clone();
    malformed.location.offset = u64::MAX;
    assert!(malformed.verify().is_err());
}
