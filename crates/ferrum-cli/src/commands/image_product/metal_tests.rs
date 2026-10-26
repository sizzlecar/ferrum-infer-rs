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
fn metal_attention_policy_is_explicit_and_shared_by_run_and_serve_configuration() {
    use ferrum_engine::image_generation::ImageMetalAttentionPolicy;
    let directory = tempfile::tempdir().unwrap();
    for serving in [false, true] {
        let mut wire = metal_wire();
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        let ImageBackendConfig::Metal(original) =
            load_wire(directory.path(), &wire).unwrap().backend
        else {
            panic!("Metal config")
        };
        assert_eq!(
            original.attention_policy,
            ImageMetalAttentionPolicy::Reference
        );
        for (value, expected) in [
            ("reference", ImageMetalAttentionPolicy::Reference),
            ("fused_dense", ImageMetalAttentionPolicy::FusedDense),
        ] {
            wire["backend"]["attention_policy"] = json!(value);
            let ImageBackendConfig::Metal(selected) =
                load_wire(directory.path(), &wire).unwrap().backend
            else {
                panic!("Metal config")
            };
            assert_eq!(selected.attention_policy, expected);
            assert_eq!(selected.precision_policy, original.precision_policy);
            assert_eq!(
                selected.host.limits().unwrap(),
                original.host.limits().unwrap()
            );
            assert_eq!(
                selected.pipeline.edit_prefix_policy,
                original.pipeline.edit_prefix_policy
            );
            #[cfg(all(feature = "metal", target_os = "macos"))]
            assert_eq!(selected.resolved().unwrap().attention_policy, expected);
        }
        for invalid in [
            json!("cudnn_sdpa_experimental_v1"),
            json!("fast"),
            json!(null),
            json!({"kind":"fused_dense"}),
        ] {
            wire["backend"]["attention_policy"] = invalid;
            assert!(load_wire(directory.path(), &wire).is_err());
        }
    }
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
    wire["backend"]["pipeline"]["edit_prefix_policy"] = json!("unknown_cache");
    assert!(load_wire(directory.path(), &wire).is_err());
}

#[test]
fn metal_request_cache_is_explicit_and_preserves_run_and_serve_limits() {
    use ferrum_engine::image_generation::{
        ImageDynamicResidencyPolicy, ImageEditPrefixPolicy, ImageResidencyPolicy,
    };
    let directory = tempfile::tempdir().unwrap();
    for serving in [false, true] {
        let mut wire = metal_wire();
        if serving {
            wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
        }
        let ImageBackendConfig::Metal(default) =
            load_wire(directory.path(), &wire).unwrap().backend
        else {
            panic!("Metal config")
        };
        assert_eq!(
            default.pipeline.edit_prefix_policy,
            ImageEditPrefixPolicy::Recompute
        );
        wire["backend"]["pipeline"]["edit_prefix_policy"] = json!("request_cache");
        let ImageBackendConfig::Metal(cached) = load_wire(directory.path(), &wire).unwrap().backend
        else {
            panic!("Metal config")
        };
        let limits = cached.pipeline.resolved_limits().unwrap();
        assert_eq!(
            limits.edit_prefix_policy,
            ImageEditPrefixPolicy::RequestCache
        );
        assert_eq!(limits.residency, ImageResidencyPolicy::Cold);
        assert_eq!(
            limits.dynamic_residency,
            ImageDynamicResidencyPolicy::Release
        );
        assert_eq!(
            cached.host.limits().unwrap(),
            default.host.limits().unwrap()
        );
        assert_eq!(
            cached.maximum_commands_per_submission,
            default.maximum_commands_per_submission
        );
        assert_eq!(
            cached.bootstrap_memory_bytes,
            default.bootstrap_memory_bytes
        );
        assert_eq!(cached.stream_memory_bytes, default.stream_memory_bytes);
        match (cached.memory, default.memory) {
            (
                configuration::MetalMemoryConfig::AutoRecommended {
                    utilization: a,
                    maximum_bytes: ac,
                },
                configuration::MetalMemoryConfig::AutoRecommended {
                    utilization: b,
                    maximum_bytes: bc,
                },
            ) => assert_eq!((a, ac), (b, bc)),
            _ => panic!("auto memory policy must remain unchanged"),
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let resolved = cached.resolved().unwrap();
            assert_eq!(
                resolved.pipeline.edit_prefix_policy,
                ImageEditPrefixPolicy::RequestCache
            );
            assert_eq!(resolved.pipeline.residency, ImageResidencyPolicy::Cold);
            assert_eq!(
                resolved.maximum_commands_per_submission,
                cached.maximum_commands_per_submission
            );
        }
    }
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
        ] {
            let mut invalid = args.clone();
            invalid.extend(extra);
            assert!(validate(&invalid).is_err(), "accepted {extra:?}");
        }
        let mut fp8 = args.clone();
        fp8.extend(["--image-precision", "fp8_linear_experimental_v1"]);
        assert!(validate(&fp8).is_ok());
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
