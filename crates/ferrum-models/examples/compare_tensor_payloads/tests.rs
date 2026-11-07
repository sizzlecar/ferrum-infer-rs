use super::*;
use safetensors::tensor::{serialize_to_file, TensorView};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn metrics_use_reference_norm_and_report_exact_worst_coordinate() {
    let left = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let reference = f32_bytes(&[1.0, 2.0, 3.0, 3.0]);
    let m = compare(&left, &reference, Dtype::F32, Dtype::F32, &[2, 2]).unwrap();
    assert!(m.finite);
    assert_eq!(m.max_abs, 1.0);
    assert_eq!(m.rmse, 0.5);
    assert!((m.relative_l2.unwrap() - 1.0 / 23.0_f64.sqrt()).abs() < 1e-15);
    assert!((m.cosine.unwrap() - 26.0 / (30.0_f64.sqrt() * 23.0_f64.sqrt())).abs() < 1e-15);
    assert_eq!(m.worst.flat_index, 3);
    assert_eq!(m.worst.coordinate, [1, 1]);
    assert_eq!(m.worst.left, 4.0);
    assert_eq!(m.worst.reference, 3.0);
    let zeros = f32_bytes(&[0.0, -0.0]);
    let m = compare(&zeros, &zeros, Dtype::F32, Dtype::F32, &[2]).unwrap();
    assert_eq!(m.max_abs, 0.0);
    assert_eq!(m.rmse, 0.0);
    assert!(m.relative_l2.is_none() && m.cosine.is_none());
    let m = compare(
        &f32_bytes(&[1.0, 0.0]),
        &zeros,
        Dtype::F32,
        Dtype::F32,
        &[2],
    )
    .unwrap();
    assert!(m.relative_l2.is_none() && m.cosine.is_none());
}

#[test]
fn real_files_preserve_reference_dtype_payload_hash_and_combined_limit() {
    let directory = tempfile::tempdir().unwrap();
    let values = [1.0, -2.0, 3.0, 4.0];
    let left = f32_bytes(&values);
    let left_path = directory.path().join("native.raw");
    std::fs::write(&left_path, &left).unwrap();
    for dtype in [Dtype::F32, Dtype::BF16, Dtype::F16] {
        let payload = match dtype {
            Dtype::F32 => left.clone(),
            Dtype::BF16 => values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            Dtype::F16 => values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            _ => unreachable!(),
        };
        let reference = directory.path().join("reference.safetensors");
        serialize_to_file(
            [(
                "features",
                TensorView::new(dtype, vec![2, 2], &payload).unwrap(),
            )],
            &None,
            &reference,
        )
        .unwrap();
        let file = std::fs::read(&reference).unwrap();
        let mut args = Arguments {
            left: LeftInput::Raw {
                path: left_path.clone(),
                shape: vec![2, 2],
            },
            group_axis: None,
            reference,
            tensor: "features".into(),
            maximum_input_bytes: left.len() + file.len(),
        };
        let report = run(&args).unwrap();
        assert_eq!(report["reference"]["dtype"], format!("{dtype:?}"));
        assert_eq!(report["reference"]["payload_sha256"], sha256(&payload));
        assert_eq!(report["reference"]["file_sha256"], sha256(&file));
        assert_eq!(report["left"]["payload_sha256"], sha256(&left));
        assert!(report.get("groups").is_none());
        assert!(report["left"].get("tensor").is_none());
        assert_eq!(report["metrics"]["max_abs"], 0.0);
        assert_eq!(report["metrics"]["rmse"], 0.0);
        assert!((report["metrics"]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-15);
        args.maximum_input_bytes -= 1;
        assert!(run(&args).is_err());
        args.maximum_input_bytes += 1;
        args.left = LeftInput::Raw {
            path: left_path.clone(),
            shape: vec![4],
        };
        assert!(run(&args)
            .unwrap_err()
            .to_string()
            .contains("shape mismatch"));
        args.left = LeftInput::Raw {
            path: left_path.clone(),
            shape: vec![2, 2],
        };
        args.tensor = "missing".into();
        assert!(run(&args).is_err());
    }
    std::fs::write(&left_path, &left[..15]).unwrap();
    assert!(read_bounded(&left_path, 16, Some(16)).is_err());
    assert!(read_bounded(directory.path(), 16, None).is_err());
}

#[test]
fn invalid_lengths_shapes_dtypes_and_nonfinite_values_are_rejected() {
    assert!(elements(&[]).is_err());
    assert!(elements(&[2, 0]).is_err());
    assert!(elements(&[usize::MAX, 2]).is_err());
    let finite = f32_bytes(&[1.0]);
    assert!(compare(&finite[..3], &finite, Dtype::F32, Dtype::F32, &[1]).is_err());
    assert!(compare(&finite, &[0], Dtype::F32, Dtype::F16, &[1]).is_err());
    assert!(compare(&finite, &finite, Dtype::F32, Dtype::F32, &[2]).is_err());
    assert!(compare(&finite, &[0], Dtype::F32, Dtype::U8, &[1]).is_err());
    assert!(compare(&[0], &finite, Dtype::U8, Dtype::F32, &[1]).is_err());
    for dtype in [Dtype::BF16, Dtype::F16] {
        assert!(compare(&[0], &finite, dtype, Dtype::F32, &[1]).is_err());
        assert!(compare(&[0; 2], &finite, dtype, Dtype::F32, &[2]).is_err());
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(compare(&f32_bytes(&[value]), &finite, Dtype::F32, Dtype::F32, &[1]).is_err());
        assert!(compare(&finite, &f32_bytes(&[value]), Dtype::F32, Dtype::F32, &[1]).is_err());
        for (dtype, bits) in [
            (Dtype::F16, f16::from_f32(value).to_bits()),
            (Dtype::BF16, bf16::from_f32(value).to_bits()),
        ] {
            assert!(compare(&finite, &bits.to_le_bytes(), Dtype::F32, dtype, &[1]).is_err());
            assert!(compare(&bits.to_le_bytes(), &finite, dtype, Dtype::F32, &[1]).is_err());
        }
    }
}

#[test]
fn cli_requires_explicit_shape_bound_and_rejects_duplicate_options() {
    let base = [
        "--left-raw",
        "native.raw",
        "--reference-safetensors",
        "reference.safetensors",
        "--tensor",
        "features",
        "--left-shape",
        "2,2",
        "--maximum-input-bytes",
        "4096",
    ];
    assert!(Arguments::parse(base.map(OsString::from))
        .unwrap()
        .is_some());
    assert!(Arguments::parse(base[..8].iter().map(|s| OsString::from(*s))).is_err());
    let mut repeated = base.map(OsString::from).to_vec();
    repeated.extend(["--tensor", "features"].map(OsString::from));
    assert!(Arguments::parse(repeated).is_err());
    for (position, value) in [(7, "2,0"), (7, "2,,2"), (9, "0"), (9, "15")] {
        let mut args = base.map(OsString::from);
        args[position] = value.into();
        assert!(Arguments::parse(args).is_err());
    }
}

#[test]
fn two_safetensors_select_distinct_names_and_group_original_coordinates() {
    let root = tempfile::tempdir().unwrap();
    let left_path = root.path().join("left.safetensors");
    let reference_path = root.path().join("reference.safetensors");
    let reference = f32_bytes(&[1.0; 12]);
    let mut values = [1.0; 12];
    values[2] += 2.0;
    values[8] += 4.0;
    values[6] += 3.0;
    let left = f32_bytes(&values);
    serialize_to_file(
        [(
            "connector_input",
            TensorView::new(Dtype::F32, vec![2, 3, 2], &left).unwrap(),
        )],
        &None,
        &left_path,
    )
    .unwrap();
    serialize_to_file(
        [(
            "hidden_states",
            TensorView::new(Dtype::F32, vec![2, 3, 2], &reference).unwrap(),
        )],
        &None,
        &reference_path,
    )
    .unwrap();
    let left_file = std::fs::read(&left_path).unwrap();
    let right_file = std::fs::read(&reference_path).unwrap();
    let mut args = Arguments {
        left: LeftInput::Safetensors {
            path: left_path.clone(),
            tensor: "connector_input".into(),
        },
        reference: reference_path.clone(),
        tensor: "hidden_states".into(),
        maximum_input_bytes: left_file.len() + right_file.len(),
        group_axis: Some(1),
    };
    let result = run(&args).unwrap();
    assert_eq!(result["left"]["tensor"], "connector_input");
    assert_eq!(result["reference"]["tensor"], "hidden_states");
    assert_eq!(result["left"]["payload_sha256"], sha256(&left));
    assert_eq!(result["left"]["file_sha256"], sha256(&left_file));
    assert_eq!(result["groups"].as_array().unwrap().len(), 3);
    assert_eq!(result["groups"][0]["metrics"]["rmse"], 1.5);
    assert!(
        (result["groups"][1]["metrics"]["rmse"].as_f64().unwrap() - 5.0_f64.sqrt()).abs() < 1e-15
    );
    assert_eq!(result["groups"][1]["metrics"]["worst"]["flat_index"], 8);
    assert_eq!(
        result["groups"][1]["metrics"]["worst"]["coordinate"],
        serde_json::json!([1, 1, 0])
    );
    assert_eq!(result["groups"][2]["metrics"]["max_abs"], 0.0);
    assert!((result["metrics"]["rmse"].as_f64().unwrap() - (29.0_f64 / 12.0).sqrt()).abs() < 1e-15);
    args.group_axis = Some(0);
    let first = run(&args).unwrap();
    assert_eq!(first["groups"].as_array().unwrap().len(), 2);
    assert_eq!(first["groups"][0]["elements"], 6);
    assert_eq!(first["groups"][0]["metrics"]["max_abs"], 2.0);
    assert_eq!(first["groups"][1]["metrics"]["max_abs"], 4.0);
    args.group_axis = Some(2);
    let last = run(&args).unwrap();
    assert_eq!(last["groups"][0]["metrics"]["max_abs"], 4.0);
    assert_eq!(last["groups"][1]["metrics"]["max_abs"], 0.0);
    args.group_axis = Some(3);
    assert!(run(&args).unwrap_err().to_string().contains("group axis"));
    args.group_axis = Some(1);
    args.maximum_input_bytes -= 1;
    assert!(run(&args).is_err());
    args.maximum_input_bytes += 1;
    args.left = LeftInput::Safetensors {
        path: left_path.clone(),
        tensor: "missing".into(),
    };
    assert!(run(&args).is_err());
    args.left = LeftInput::Safetensors {
        path: left_path.clone(),
        tensor: "connector_input".into(),
    };
    serialize_to_file(
        [(
            "hidden_states",
            TensorView::new(Dtype::F32, vec![2, 6], &reference).unwrap(),
        )],
        &None,
        &reference_path,
    )
    .unwrap();
    assert!(run(&args)
        .unwrap_err()
        .to_string()
        .contains("shape mismatch"));
    let half = [0u8; 24];
    serialize_to_file(
        [(
            "connector_input",
            TensorView::new(Dtype::U16, vec![2, 3, 2], &half).unwrap(),
        )],
        &None,
        &left_path,
    )
    .unwrap();
    assert!(format!("{:#}", run(&args).unwrap_err()).contains("unsupported tensor dtype U16"));
}

#[test]
fn half_safetensors_keep_each_dtype_and_group_stride_without_changing_metrics() {
    let root = tempfile::tempdir().unwrap();
    let left_path = root.path().join("left.safetensors");
    let reference_path = root.path().join("reference.safetensors");
    let encode = |values: &[f32], dtype| -> Vec<u8> {
        match dtype {
            Dtype::F32 => f32_bytes(values),
            Dtype::BF16 => values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            Dtype::F16 => values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            _ => unreachable!(),
        }
    };
    for (left_dtype, reference_dtype) in [
        (Dtype::BF16, Dtype::F16),
        (Dtype::BF16, Dtype::BF16),
        (Dtype::F16, Dtype::F32),
    ] {
        let left = encode(&[1.0, -2.0, 3.0, 4.0], left_dtype);
        let reference = encode(&[1.0, -2.0, 3.0, 3.5], reference_dtype);
        serialize_to_file(
            [(
                "left",
                TensorView::new(left_dtype, vec![2, 2], &left).unwrap(),
            )],
            &None,
            &left_path,
        )
        .unwrap();
        serialize_to_file(
            [(
                "right",
                TensorView::new(reference_dtype, vec![2, 2], &reference).unwrap(),
            )],
            &None,
            &reference_path,
        )
        .unwrap();
        let left_file = std::fs::read(&left_path).unwrap();
        let reference_file = std::fs::read(&reference_path).unwrap();
        let report = run(&Arguments {
            left: LeftInput::Safetensors {
                path: left_path.clone(),
                tensor: "left".into(),
            },
            group_axis: Some(1),
            reference: reference_path.clone(),
            tensor: "right".into(),
            maximum_input_bytes: left_file.len() + reference_file.len(),
        })
        .unwrap();
        assert_eq!(report["schema_version"], 1);
        assert_eq!(report["normalization"], "reference_l2");
        assert_eq!(report["left"]["dtype"], format!("{left_dtype:?}"));
        assert_eq!(report["reference"]["dtype"], format!("{reference_dtype:?}"));
        assert_eq!(report["left"]["payload_sha256"], sha256(&left));
        assert_eq!(report["left"]["file_sha256"], sha256(&left_file));
        assert_eq!(report["reference"]["payload_sha256"], sha256(&reference));
        assert_eq!(report["reference"]["file_sha256"], sha256(&reference_file));
        assert_eq!(report["metrics"]["max_abs"], 0.5);
        assert_eq!(report["metrics"]["rmse"], 0.25);
        assert!(
            (report["metrics"]["relative_l2"].as_f64().unwrap() - 0.5 / 26.25_f64.sqrt()).abs()
                < 1e-15
        );
        assert!(
            (report["metrics"]["cosine"].as_f64().unwrap()
                - 28.0 / (30.0_f64.sqrt() * 26.25_f64.sqrt()))
            .abs()
                < 1e-15
        );
        assert_eq!(report["groups"][0]["metrics"]["max_abs"], 0.0);
        assert_eq!(report["groups"][1]["metrics"]["max_abs"], 0.5);
        assert_eq!(
            report["groups"][1]["metrics"]["worst"]["coordinate"],
            serde_json::json!([1, 1])
        );
        assert!(
            (report["groups"][1]["metrics"]["relative_l2"]
                .as_f64()
                .unwrap()
                - 0.5 / 16.25_f64.sqrt())
            .abs()
                < 1e-15
        );
    }
}

#[test]
fn safetensors_left_cli_requires_its_own_name_and_rejects_raw_shape_combinations() {
    let base = [
        "--left-safetensors",
        "native.safetensors",
        "--left-tensor",
        "connector_input",
        "--reference-safetensors",
        "reference.safetensors",
        "--tensor",
        "hidden_states",
        "--maximum-input-bytes",
        "134217728",
        "--group-axis",
        "1",
    ];
    let args = Arguments::parse(base.map(OsString::from)).unwrap().unwrap();
    assert_eq!(args.group_axis, Some(1));
    assert!(matches!(&args.left,LeftInput::Safetensors { tensor,.. } if tensor=="connector_input"));
    for extra in [
        ["--left-raw", "native.raw"],
        ["--left-shape", "1,5,1180,2560"],
    ] {
        let all = base.into_iter().chain(extra).map(OsString::from);
        assert!(Arguments::parse(all).is_err());
    }
    let missing_name = base[..2]
        .iter()
        .chain(base[4..].iter())
        .map(|value| OsString::from(*value));
    assert!(Arguments::parse(missing_name).is_err());
    for value in ["-1", "1.5", "invalid"] {
        let mut args = base;
        args[11] = value;
        assert!(Arguments::parse(args.map(OsString::from)).is_err());
    }
    let mut empty = base;
    empty[3] = "";
    assert!(Arguments::parse(empty.map(OsString::from)).is_err());
}
