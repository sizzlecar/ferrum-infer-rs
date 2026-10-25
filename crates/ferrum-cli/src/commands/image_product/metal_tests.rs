use super::*;
use crate::commands::image_product::directory::ImageBackendKind;

fn metal_wire() -> Value {
    let mut wire = configuration();
    let backend = wire["backend"].as_object_mut().unwrap();
    for key in ["ordinal", "usable_memory_bytes", "cudnn"] {
        backend.remove(key);
    }
    backend.insert("kind".into(), json!("metal"));
    backend.insert(
        "memory".into(),
        json!({"kind":"auto_recommended","utilization":0.75,"maximum_bytes":67108864}),
    );
    wire
}
fn load_wire(directory: &std::path::Path, wire: &Value) -> anyhow::Result<ImageProductConfig> {
    let path = directory.join("metal.json");
    fs::write(&path, serde_json::to_vec(wire)?)?;
    ImageProductConfig::load(&path)
}

#[test]
fn metal_configuration_shares_residency_batch_limits_and_explicit_capacity() {
    let directory = tempfile::tempdir().unwrap();
    for serving in [false, true] {
        let mut wire = metal_wire();
        wire["execution_profile"] = json!("experimental_resident_v1");
        wire["backend"]["pipeline"]["residency"] = json!("all_stages");
        wire["backend"]["pipeline"]["maximum_batch_size"] = json!(2);
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":2,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        for memory in [
            json!({"kind":"auto_recommended","utilization":0.75}),
            json!({"kind":"explicit_total","bytes":67108864}),
        ] {
            wire["backend"]["memory"] = memory;
            let config = load_wire(directory.path(), &wire).unwrap();
            let ImageBackendConfig::Metal(metal) = config.backend else {
                panic!("Metal config")
            };
            metal.validate().unwrap();
            assert_eq!(metal.pipeline.maximum_batch_size, 2);
            assert_eq!(
                metal.pipeline.residency,
                ferrum_engine::image_generation::ImageResidencyPolicy::AllStages
            );
            #[cfg(all(feature = "metal", target_os = "macos"))]
            assert_eq!(metal.resolved().unwrap().pipeline.maximum_batch_size, 2);
        }
        wire["backend"]["pipeline"]["maximum_batch_size"] = json!(1);
        if serving {
            assert!(load_wire(directory.path(), &wire).is_err());
        }
        wire["execution_profile"] = json!("experimental_serial_cold");
        assert!(load_wire(directory.path(), &wire).is_err());
    }
}

#[test]
fn metal_configuration_rejects_ignored_cuda_controls_and_capacity_errors() {
    let directory = tempfile::tempdir().unwrap();
    let original = metal_wire();
    for (key, value) in [
        ("ordinal", json!(0)),
        ("cudnn", json!({})),
        ("precision_policy", json!("fp8_linear_experimental_v1")),
        ("memory", json!({"kind":"explicit_total","bytes":0})),
        ("memory", json!({"kind":"explicit_total","bytes":192})),
        (
            "memory",
            json!({"kind":"auto_recommended","utilization":0.0}),
        ),
        (
            "memory",
            json!({"kind":"auto_recommended","utilization":1.1}),
        ),
        (
            "memory",
            json!({"kind":"auto_recommended","utilization":0.75,"maximum_bytes":0}),
        ),
        ("maximum_in_flight_per_stream", json!(0)),
        ("maximum_streams", json!(1)),
        ("maximum_commands_per_submission", json!(1)),
        ("stream_memory_bytes", json!(u64::MAX)),
    ] {
        let mut wire = original.clone();
        wire["backend"][key] = value;
        assert!(
            load_wire(directory.path(), &wire).is_err(),
            "accepted {key}: {}",
            wire["backend"][key]
        );
    }
    let mut wire = original;
    wire["backend"]["pipeline"]["edit_prefix_policy"] = json!("request_cache");
    assert!(load_wire(directory.path(), &wire).is_err());
}

#[test]
fn metal_directory_routes_run_and_serve_and_validates_backend_options() {
    let root = tempfile::tempdir().unwrap();
    directory_tests::metadata_fixture(root.path());
    let model = root.path().to_str().unwrap();
    for command in ["run", "serve"] {
        let mut args = vec![
            "ferrum",
            command,
            model,
            "--backend",
            "metal",
            "--max-num-seqs",
            "2",
        ];
        if command == "run" {
            args.extend(["--prompt", "cat", "--output", "picture.png"]);
        }
        assert!(validate(&args).is_ok());
        for extra in [
            ["--image-cudnn-library", "libcudnn.so.9"],
            ["--gpu-devices", "0"],
            ["--image-precision", "fp8_linear_experimental_v1"],
        ] {
            let mut invalid = args.clone();
            invalid.extend(extra);
            assert!(validate(&invalid).is_err(), "accepted {extra:?}");
        }
        if command == "run" {
            args.extend(["--image", "source.png"]);
            assert!(validate(&args).is_ok());
        }
    }
    let options = DirectoryOptions::from_cli(
        0.75,
        std::num::NonZeroUsize::new(1024 * 1024 * 1024),
        Some(2),
        "metal",
        None,
        None,
    )
    .unwrap();
    assert_eq!(options.backend, ImageBackendKind::Metal);
    assert_eq!(
        options.maximum_memory_bytes.unwrap().get(),
        1024 * 1024 * 1024
    );
    assert_eq!(options.host().transfer_staging_bytes, 0);
    assert_eq!(
        options.serving().unwrap().max_num_seqs,
        options.pipeline().maximum_batch_size
    );
    let auto = DirectoryOptions::from_cli(0.75, None, None, "auto", None, None).unwrap();
    assert_eq!(
        auto.backend,
        if cfg!(all(feature = "metal", target_os = "macos")) {
            ImageBackendKind::Metal
        } else {
            ImageBackendKind::Cuda
        }
    );
    #[cfg(all(feature = "metal", target_os = "macos"))]
    options.metal().unwrap().validate().unwrap();
}
