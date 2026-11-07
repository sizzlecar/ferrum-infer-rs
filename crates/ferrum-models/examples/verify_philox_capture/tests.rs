use super::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

fn fixture(path: &Path, values: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    serialize_to_file(
        values.iter().map(|(name, dtype, shape, data)| {
            (*name, TensorView::new(*dtype, shape.clone(), data).unwrap())
        }),
        &None,
        path,
    )
    .unwrap();
}
fn args(path: PathBuf, output: PathBuf) -> Arguments {
    Arguments {
        capture: path,
        tensors: vec!["image".into(), "audio".into()],
        seed: 7,
        offset: 3,
        launch_blocks: 1,
        maximum_input_bytes: 16384,
        output,
    }
}
fn stream() -> Vec<f32> {
    // Construct by draw round, component, then thread. The verifier instead
    // consumes a single linear index crossing two differently shaped tensors.
    let mut values = Vec::new();
    for round in 0..2 {
        for component in 0..4 {
            for thread in 0..256 {
                values.push(oracle::normal4(PhiloxSeedOffset::new(7, 3), thread, round)[component]);
            }
        }
    }
    values.truncate(1031);
    values
}
fn stream_fixture(path: &Path, values: &[f32]) {
    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fixture(
        path,
        &[
            ("image", Dtype::F32, vec![1021], bytes(&values[..1021])),
            ("audio", Dtype::F32, vec![2, 5], bytes(&values[1021..])),
            // Must count towards the complete-file cap but not the selected stream.
            ("ignored", Dtype::U8, vec![31], vec![99; 31]),
        ],
    );
}

#[test]
fn ordered_concat_crosses_tensor_component_and_round_without_restarting_rng() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.safetensors");
    let original = stream();
    stream_fixture(&path, &original);
    let mut config = args(path, dir.path().join("report.json"));
    let report = analyze(&config).unwrap();
    assert_eq!(report.mismatch_count, 0);
    assert_eq!(report.report["elements"], 1031);
    assert_eq!(report.report["rmse"], 0.0);
    assert_eq!(report.report["selected_tensors"][1]["global_start"], 1021);
    assert_eq!(report.report["selected_tensors"][1]["global_end"], 1031);
    let sum: f64 = original.iter().map(|&v| f64::from(v)).sum();
    let expected_mean = sum / original.len() as f64;
    let expected_variance: f64 = original
        .iter()
        .map(|&v| (f64::from(v) - expected_mean).powi(2))
        .sum::<f64>()
        / original.len() as f64;
    assert!((report.report["actual"]["mean"].as_f64().unwrap() - expected_mean).abs() < 1e-12);
    assert!(
        (report.report["actual"]["population_std"].as_f64().unwrap() - expected_variance.sqrt())
            .abs()
            < 1e-12
    );
    config.tensors.reverse();
    assert!(
        analyze(&config).unwrap().mismatch_count > 0,
        "tensor order is part of the draw contract"
    );
}

#[test]
fn mismatches_keep_the_complete_report_and_identify_global_draw_position() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.safetensors");
    let mut values = stream();
    values[1024] += 1.0;
    stream_fixture(&path, &values);
    let config = args(path, dir.path().join("failed.json"));
    assert!(execute(&config).is_err());
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config.output).unwrap()).unwrap();
    assert_eq!(report["complete"], true);
    assert_eq!(report["mismatch_count"], 1);
    for key in ["first_mismatch", "maximum_error"] {
        assert_eq!(report[key]["tensor"], "audio");
        assert_eq!(report[key]["tensor_index"], 3);
        assert_eq!(report[key]["global_index"], 1024);
        assert_eq!(report[key]["thread"], 0);
        assert_eq!(report[key]["round"], 1);
        assert_eq!(report[key]["component"], 0);
    }
    assert!(report["max_abs"].as_f64().unwrap() > 0.99);
}

#[test]
fn rejects_nonfinite_wrong_precision_shape_file_cap_and_implicit_geometry() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.safetensors");
    let mut values = stream();
    stream_fixture(&path, &values);
    let mut config = args(path.clone(), dir.path().join("invalid.json"));
    config.maximum_input_bytes = path.metadata().unwrap().len() as usize - 1;
    assert!(analyze(&config).is_err());
    config.maximum_input_bytes += 1;
    assert!(analyze(&config).is_ok());
    values[0] = f32::NAN;
    stream_fixture(&path, &values);
    assert!(analyze(&config).is_err());
    config.tensors = vec!["image".into()];
    for (dtype, shape, bytes) in [
        (Dtype::BF16, vec![1], vec![0, 0]),
        (Dtype::F32, vec![0], vec![]),
    ] {
        fixture(&path, &[("image", dtype, shape, bytes)]);
        assert!(analyze(&config).is_err());
    }
    let base = [
        "--capture",
        "x.safetensors",
        "--tensor",
        "image",
        "--tensor",
        "audio",
        "--seed",
        "7",
        "--offset",
        "3",
        "--launch-blocks",
        "1",
        "--maximum-input-bytes",
        "16384",
        "--output",
        "new.json",
    ];
    let parse = |v: Vec<&str>| Arguments::parse(v.into_iter().map(OsString::from));
    assert_eq!(
        parse(base.to_vec()).unwrap().unwrap().tensors,
        ["image", "audio"]
    );
    let mut zero_blocks = base.to_vec();
    zero_blocks[11] = "0";
    assert!(parse(zero_blocks).is_err());
    let mut missing_geometry = base.to_vec();
    missing_geometry.drain(10..12);
    assert!(parse(missing_geometry).is_err());
    let mut duplicate = base.to_vec();
    duplicate.extend(["--tensor", "image"]);
    assert!(parse(duplicate).is_err());
    assert!(elements(&[u64::MAX, u64::MAX]).is_err());
    assert!(
        !config.output.exists(),
        "invalid captures do not acquire a complete report"
    );
}
