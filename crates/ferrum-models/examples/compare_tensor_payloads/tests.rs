use super::*;
use safetensors::tensor::{serialize_to_file, TensorView};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn metrics_use_reference_norm_and_report_exact_worst_coordinate() {
    let left = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let reference = f32_bytes(&[1.0, 2.0, 3.0, 3.0]);
    let m = compare(&left, &reference, Dtype::F32, &[2, 2]).unwrap();
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
    let m = compare(&zeros, &zeros, Dtype::F32, &[2]).unwrap();
    assert_eq!(m.max_abs, 0.0);
    assert_eq!(m.rmse, 0.0);
    assert!(m.relative_l2.is_none() && m.cosine.is_none());
    let m = compare(&f32_bytes(&[1.0, 0.0]), &zeros, Dtype::F32, &[2]).unwrap();
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
            left_raw: left_path.clone(),
            left_shape: vec![2, 2],
            reference,
            tensor: "features".into(),
            maximum_input_bytes: left.len() + file.len(),
        };
        let report = run(&args).unwrap();
        assert_eq!(report["reference"]["dtype"], format!("{dtype:?}"));
        assert_eq!(report["reference"]["payload_sha256"], sha256(&payload));
        assert_eq!(report["reference"]["file_sha256"], sha256(&file));
        assert_eq!(report["left"]["payload_sha256"], sha256(&left));
        assert_eq!(report["metrics"]["max_abs"], 0.0);
        assert_eq!(report["metrics"]["rmse"], 0.0);
        assert!((report["metrics"]["cosine"].as_f64().unwrap() - 1.0).abs() < 1e-15);
        args.maximum_input_bytes -= 1;
        assert!(run(&args).is_err());
        args.maximum_input_bytes += 1;
        args.left_shape = vec![4];
        assert!(run(&args)
            .unwrap_err()
            .to_string()
            .contains("shape mismatch"));
        args.left_shape = vec![2, 2];
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
    assert!(compare(&finite[..3], &finite, Dtype::F32, &[1]).is_err());
    assert!(compare(&finite, &[0], Dtype::F16, &[1]).is_err());
    assert!(compare(&finite, &finite, Dtype::F32, &[2]).is_err());
    assert!(compare(&finite, &[0], Dtype::U8, &[1]).is_err());
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(compare(&f32_bytes(&[value]), &finite, Dtype::F32, &[1]).is_err());
        assert!(compare(&finite, &f32_bytes(&[value]), Dtype::F32, &[1]).is_err());
        for (dtype, bits) in [
            (Dtype::F16, f16::from_f32(value).to_bits()),
            (Dtype::BF16, bf16::from_f32(value).to_bits()),
        ] {
            assert!(compare(&finite, &bits.to_le_bytes(), dtype, &[1]).is_err());
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
