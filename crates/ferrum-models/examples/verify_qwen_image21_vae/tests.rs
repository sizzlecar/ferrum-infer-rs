use super::*;
use arguments::{LatentDomain, LatentLayout};
use ferrum_models::qwen_image21::QwenImage21VaeDecoderConfig as Config;
use half::bf16;
use std::{collections::BTreeMap, ffi::OsString};

fn config(channels: u64, batch: u64, height: u64, width: u64) -> Config {
    Config::new(channels, 1, vec![1], 0, vec![], batch, height, width).unwrap()
}
fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| bf16::from_f32(x).to_bits().to_le_bytes())
        .collect()
}
fn bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect()
}
fn cli() -> Vec<OsString> {
    [
        "--model-root",
        "model",
        "--latent-file",
        "latent.bf16",
        "--latent-layout",
        "packed",
        "--latent-domain",
        "normalized",
        "--latent-height",
        "1",
        "--latent-width",
        "2",
        "--batch",
        "2",
        "--artifact-dir",
        "out",
        "--backend",
        "reference",
    ]
    .map(OsString::from)
    .to_vec()
}
#[test]
fn explicit_domain_layout_backend_and_cudnn_budgets_are_required() {
    let a = Arguments::parse(cli()).unwrap().unwrap();
    assert_eq!(a.config, std::path::PathBuf::from("model/vae/config.json"));
    assert_eq!((a.batch, a.height, a.width), (2, 1, 2));
    for flag in ["--latent-layout", "--latent-domain", "--backend"] {
        let mut args = cli();
        let i = args.iter().position(|v| v == flag).unwrap();
        args.drain(i..i + 2);
        assert!(Arguments::parse(args).is_err());
    }
    for (flag, value) in [
        ("--latent-layout", "patched"),
        ("--latent-domain", "auto"),
        ("--latent-height", "0"),
        ("--batch", "0"),
    ] {
        let mut args = cli();
        let i = args.iter().position(|v| v == flag).unwrap();
        args[i + 1] = value.into();
        assert!(Arguments::parse(args).is_err());
    }
    let mut ref_cudnn = cli();
    ref_cudnn.extend(["--cudnn-library", "/explicit/lib.so"].map(OsString::from));
    assert!(Arguments::parse(ref_cudnn).is_err());
    let mut cuda = cli();
    *cuda.last_mut().unwrap() = "cuda".into();
    cuda.extend(
        [
            "--cuda-usable-memory-bytes",
            "2147483648",
            "--cuda-bootstrap-memory-bytes",
            "1073741824",
            "--cuda-stream-memory-bytes",
            "67108864",
            "--cuda-opaque-memory-profile",
            "experimental-test",
            "--cudnn-library",
            "/explicit/lib.so",
            "--cudnn-opaque-memory-bytes",
            "67108864",
            "--cudnn-workspace-bytes",
            "0",
            "--cudnn-memory-profile",
            "experimental-cudnn",
        ]
        .map(OsString::from),
    );
    assert!(Arguments::parse(cuda.clone())
        .unwrap()
        .unwrap()
        .cudnn
        .is_some());
    let i = cuda.iter().position(|v| v == "--cudnn-library").unwrap();
    cuda[i + 1] = "relative.so".into();
    assert!(Arguments::parse(cuda).is_err());
}
#[test]
fn packed_latent_is_unpatched_cross_batch_transpose_with_one_backing() {
    let c = config(2, 2, 1, 2);
    let budget = latent::Budget::new(&c, 2, 50).unwrap();
    assert_eq!(budget.total, 50);
    assert!(latent::Budget::new(&c, 2, 49).is_err());
    let host = budget.account().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("packed");
    let source = bytes(&[1., 10., 2., 20., 3., 30., 4., 40.]);
    std::fs::write(&path, &source).unwrap();
    let loaded = latent::load(
        &path,
        &c,
        LatentLayout::Packed,
        LatentDomain::Denormalized,
        None,
        &host,
    )
    .unwrap();
    assert_eq!(
        loaded.tensor.bytes().bytes(),
        bytes(&[1., 2., 10., 20., 3., 4., 30., 40.])
    );
    assert_eq!(loaded.source_sha256, artifacts::sha256(&source));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 16);
    let alias = loaded.tensor.bytes().slice(2..4).unwrap();
    drop(loaded);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 16);
    drop(alias);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
#[test]
fn normalized_latent_has_two_bf16_boundaries_and_errors_release_backing() {
    let c = config(1, 1, 1, 2);
    let budget = latent::Budget::new(&c, 2, 64).unwrap();
    let host = budget.account().unwrap();
    let stats = latent::Statistics::from_config(
        &json!({"latents_std":[129./128.],"latents_mean":[-130./128.]}),
        1,
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("latent");
    std::fs::write(&path, bytes(&[129. / 128., 129. / 128.])).unwrap();
    let result = latent::load(
        &path,
        &c,
        LatentLayout::Nchw,
        LatentDomain::Normalized,
        Some(&stats),
        &host,
    )
    .unwrap();
    assert_eq!(bits(result.tensor.bytes().bytes()), [0, 0]);
    assert_eq!(
        bf16::from_f32((129_f32 / 128.) * (129_f32 / 128.) - 130_f32 / 128.).to_bits(),
        0x3880
    );
    drop(result);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    std::fs::write(&path, [0x80, 0x3f, 0x80, 0x7f]).unwrap();
    assert!(latent::load(
        &path,
        &c,
        LatentLayout::Nchw,
        LatentDomain::Normalized,
        Some(&stats),
        &host
    )
    .is_err());
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    std::fs::write(&path, [0, 0]).unwrap();
    assert!(latent::load(
        &path,
        &c,
        LatentLayout::Nchw,
        LatentDomain::Denormalized,
        None,
        &host
    )
    .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    for bad in [
        json!({"latents_std":[0],"latents_mean":[0]}),
        json!({"latents_std":[1],"latents_mean":[]}),
    ] {
        assert!(latent::Statistics::from_config(&bad, 1).is_err());
    }
}
#[test]
fn pixel_mapping_keeps_rgba_and_bf16_shift_boundary() {
    let values = [-2., -1., -0.5, 0., 0.5, 1., 2.];
    let expected = [0, 0, 64, 128, 191, 255, 255];
    for (x, y) in values.into_iter().zip(expected) {
        assert_eq!(output::rgba8(bf16::from_f32(x)).unwrap(), y);
    }
    // The BF16 shift rounds down to 0.50390625; omitting it crosses an
    // integer-pixel rounding midpoint and produces 129 instead of 128.
    let boundary = bf16::from_bits(0x3c01);
    assert_eq!(output::rgba8(boundary).unwrap(), 128);
    assert_eq!(
        ((boundary.to_f32() * 0.5 + 0.5) * 255.0).round_ties_even(),
        129.0
    );
    assert!(output::rgba8(bf16::NAN).is_err());
}
#[test]
fn tiny_real_archive_runs_full_decoder_and_writes_raw_rgba_and_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    let vae = model.join("vae");
    std::fs::create_dir_all(&vae).unwrap();
    let c = config(1, 1, 1, 1);
    let raw = json!({"_class_name":"AutoencoderKLQwenImage21","is_residual":true,"patch_size":null,"dropout":0.0,"out_channels":4,"attn_scales":[],"base_dim":1,"decoder_base_dim":1,"z_dim":1,"dim_mult":[1],"num_res_blocks":0,"temperal_downsample":[],"latents_std":[1.0],"latents_mean":[0.0]});
    std::fs::write(vae.join("config.json"), serde_json::to_vec(&raw).unwrap()).unwrap();
    let mut payloads = BTreeMap::new();
    for w in c.expected_weights().unwrap() {
        let mut data = vec![0_u8; w.bytes() as usize];
        if w.external_name().ends_with(".gamma") {
            for pair in data.chunks_exact_mut(2) {
                pair.copy_from_slice(&0x3f80_u16.to_le_bytes());
            }
        }
        payloads.insert(
            w.external_name().to_string(),
            (
                w.dimensions()
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>(),
                data,
            ),
        );
    }
    let views = payloads
        .iter()
        .map(|(n, (s, b))| {
            (
                n.as_str(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, s.clone(), b)
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(views, &None, &vae.join("model.safetensors")).unwrap();
    let latent = dir.path().join("latent");
    std::fs::write(&latent, [0x80, 0x3f]).unwrap();
    let out = dir.path().join("output");
    let args = Arguments {
        model_root: model,
        config: vae.join("config.json"),
        index_file: None,
        latent_file: latent,
        layout: LatentLayout::Packed,
        domain: LatentDomain::Normalized,
        height: 1,
        width: 1,
        batch: 1,
        artifact_dir: out.clone(),
        maximum_host_bytes: 1 << 20,
        reference_memory_bytes: 64 << 20,
        backend: backend::Backend::Reference,
        cudnn: None,
    };
    run(args).unwrap();
    assert_eq!(std::fs::read(out.join("rgba.bf16")).unwrap(), [0; 8]);
    assert_eq!(std::fs::read(out.join("rgba-0.rgba8")).unwrap(), [128; 4]);
    let result: Value =
        serde_json::from_slice(&std::fs::read(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["validation"]["passed"], true);
    assert_eq!(
        result["validation"]["png"][0]["decoded_shape"],
        json!([1, 1, 4])
    );
    assert_eq!(
        result["validation"]["png"][0]["decoded_rgba8_sha256"],
        result["validation"]["rgba8"][0]["sha256"]
    );
    let png = std::fs::read(out.join("rgba-0.png")).unwrap();
    assert_eq!(
        result["validation"]["png"][0]["sha256"],
        artifacts::sha256(&png)
    );
    assert_eq!(result["cleanup"]["managed_host_committed_bytes"], 0);
    assert_eq!(result["cleanup"]["released_static_resources"], 1);
    let identity: Value =
        serde_json::from_slice(&std::fs::read(out.join("declaration.json")).unwrap()).unwrap();
    assert_eq!(identity["latent"]["domain"], "normalized");
    assert!(!identity["weights"].as_array().unwrap().is_empty());
}
