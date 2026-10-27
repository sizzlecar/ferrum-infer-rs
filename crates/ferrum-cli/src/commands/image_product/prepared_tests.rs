use super::*;
use ferrum_engine::image_generation::ImageStage;

#[path = "metal_integration/assets.rs"]
mod assets;

#[test]
fn cpu_prepare_command_exports_both_stages_and_shared_asset_loader_binds_them() {
    use crate::commands::prepare_image_weights::{
        execute, PrepareImageWeightsCommand, PreparedImageStage,
    };
    use sha2::{Digest, Sha256};
    let directory = assets::package();
    let root = directory.path();
    let path = root.join("image.json");
    let mut wire = configuration();
    wire["metadata"] = json!({"model_index":"model_index.json", "text_encoder":"text_encoder/config.json",
        "transformer":"transformer/config.json", "vae":"vae/config.json", "scheduler":"scheduler/scheduler_config.json",
        "tokenizer":"processor/tokenizer.json", "tokenizer_config":"processor/tokenizer_config.json",
        "chat_template":"processor/chat_template.jinja", "processor":"processor/processor_config.json"});
    wire["weights"] = json!({"text_encoder":{"kind":"single","directory":"text_encoder","file":"model.safetensors"},
        "transformer":{"kind":"single","directory":"transformer","file":"diffusion_pytorch_model.safetensors"},
        "vae":{"kind":"single","directory":"vae","file":"diffusion_pytorch_model.safetensors"}});
    wire["backend"]["precision_policy"] = json!("fp8_linear_experimental_v1");
    wire["backend"]["host"] = json!({"total_bytes":8388608,"input_preprocess_bytes":1048576,
        "transfer_staging_bytes":1048576,"output_bytes":1048576,"weight_backing_bytes":4194304});
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    let original = ImageProductConfig::load(&path)
        .unwrap()
        .prepare_assets()
        .unwrap();
    let mut selected = json!({});
    for (name, stage) in [
        ("text_encoder", PreparedImageStage::TextEncoder),
        ("denoiser", PreparedImageStage::Denoiser),
    ] {
        let output = root.join(format!("prepared-{name}"));
        execute(PrepareImageWeightsCommand {
            image_config: path.clone(),
            stage,
            output: output.clone(),
        })
        .unwrap();
        let manifest = output.join("prepared-fp8.json");
        let digest = format!("{:x}", Sha256::digest(fs::read(&manifest).unwrap()));
        selected[name] = json!({"manifest":manifest, "sha256":digest});
        assert!(execute(PrepareImageWeightsCommand {
            image_config: path.clone(),
            stage,
            output
        })
        .is_err());
    }
    wire["prepared_weights"] = selected;
    for serving in [false, true] {
        wire["serving"] = if serving {
            json!({"maximum_pending":2,"maximum_prompt_bytes":4096,"max_num_seqs":1,
                "request_batch_max_wait_ms":0,"default_width":512,"default_height":512,
                "default_seed":0,"rgba_scratch_bytes":16384,"output":{}})
        } else {
            Value::Null
        };
        fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
        let assets = ImageProductConfig::load(&path)
            .unwrap()
            .prepare_assets()
            .unwrap();
        assert_eq!(
            assets.receipt().required_weight_bytes,
            original.receipt().required_weight_bytes
        );
        for (name, key) in [
            ("text_encoder", "prepared_text_encoder_manifest"),
            ("denoiser", "prepared_denoiser_manifest"),
        ] {
            assert_eq!(
                assets.receipt().metadata_sha256[key],
                wire["prepared_weights"][name]["sha256"].as_str().unwrap()
            );
        }
    }
    wire["prepared_weights"]["text_encoder"]["sha256"] = json!("0".repeat(64));
    fs::write(&path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(ImageProductConfig::load(&path)
        .unwrap()
        .prepare_assets()
        .is_err());
}

#[test]
fn prepared_weights_require_explicit_identity_and_matching_run_serve_precision() {
    let directory = tempfile::tempdir().unwrap();
    let config_path = directory.path().join("image.json");
    let load = |wire: &Value| {
        fs::write(&config_path, serde_json::to_vec(wire).unwrap()).unwrap();
        ImageProductConfig::load(&config_path)
    };
    for metal in [false, true] {
        for serving in [false, true] {
            let mut wire = configuration();
            if metal {
                let backend = wire["backend"].as_object_mut().unwrap();
                for key in ["ordinal", "usable_memory_bytes", "cudnn"] {
                    backend.remove(key);
                }
                backend.insert("kind".into(), json!("metal"));
                backend.insert(
                    "memory".into(),
                    json!({"kind":"explicit_total", "bytes":67108864}),
                );
            }
            if serving {
                wire["serving"] = json!({"maximum_pending":2,"maximum_prompt_bytes":4096,
                    "max_num_seqs":1,"request_batch_max_wait_ms":0,"default_width":512,
                    "default_height":512,"default_seed":0,"rgba_scratch_bytes":16384,"output":{}});
            }
            assert_eq!(load(&wire).unwrap().prepared_weights.entries().count(), 0);
            wire["prepared_weights"] = json!({"text_encoder":{
                "manifest":"prepared/text/prepared-fp8.json", "sha256":"a".repeat(64)}});
            assert!(
                load(&wire).is_err(),
                "BF16 policy must not silently ignore prepared FP8"
            );
            wire["backend"]["precision_policy"] = json!("fp8_linear_experimental_v1");
            wire["backend"]["host"] = json!({"total_bytes":5120,"input_preprocess_bytes":1024,
                "transfer_staging_bytes":1024,"output_bytes":2048,"weight_backing_bytes":1024});
            let selected = load(&wire).unwrap();
            let entries = selected.prepared_weights.entries().collect::<Vec<_>>();
            assert_eq!(entries.len(), 1);
            assert_eq!(entries[0].0, ImageStage::TextEncoder);
            assert_eq!(
                entries[0].1.manifest,
                directory
                    .path()
                    .canonicalize()
                    .unwrap()
                    .join("prepared/text/prepared-fp8.json")
            );
            for digest in ["", "bad", &"g".repeat(64), &"A".repeat(64)] {
                let mut invalid = wire.clone();
                invalid["prepared_weights"]["text_encoder"]["sha256"] = json!(digest);
                assert!(load(&invalid).is_err());
            }
            for unknown in ["vae", "automatic_cache"] {
                let mut invalid = wire.clone();
                invalid["prepared_weights"][unknown] = json!({});
                assert!(load(&invalid).is_err());
            }
            if !metal {
                wire["backend"]["fp8_host_cache"] = json!({"stage_budgets":{
                    "maximum_bytes_by_stage":{"text_encoder":1024}}});
                assert!(
                    load(&wire).is_err(),
                    "do not retain a duplicate Host FP8 cache"
                );
            }
        }
    }
}
