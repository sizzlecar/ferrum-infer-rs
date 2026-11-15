use super::*;
mod blocks;
mod clip;
#[cfg(feature = "cuda-cutlass-nvfp4")]
mod cuda_support;
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use std::collections::{BTreeMap, HashMap};

fn config_json() -> serde_json::Value {
    serde_json::json!({
        "_class_name":"AutoencoderKLMiniMaxH3", "_diffusers_version":"0.36.0.dev0",
        "in_channels":3,"out_channels":3,"latent_channels":2,
        "block_out_channels":[32,64],"layers_per_block":2,
        "spatial_downsample_factors":[2,1],"temporal_downsample_factors":[2,2],
        "norm_num_groups":32,"norm_eps":1e-6,"spatial_padding_mode":"reflect",
        "decoder_num_layers":1,"decoder_num_attention_heads":2,"decoder_attention_head_dim":64,
        "decoder_num_register_tokens":4,"decoder_ffn_mult":2,
        "decoder_rope_theta":100.0,"decoder_rope_dim_ratio":0.75,"decoder_norm_eps":1e-5,
        "clip_length":17,"token_drop":3,"latents_mean":[0.25,-0.5],"latents_std":[1.0,2.0]
    })
}
pub(super) fn config() -> FastH3VideoVaeConfig {
    FastH3VideoVaeConfig::parse(&serde_json::to_vec(&config_json()).unwrap(), 4096).unwrap()
}
fn limits() -> FastH3VideoDecoderWeightLimits {
    FastH3VideoDecoderWeightLimits {
        maximum_tensors: 64,
        maximum_source_bytes: 1 << 20,
        maximum_metadata_bytes: 1024,
    }
}

#[test]
fn temporal_decode_matches_clip_pattern_and_preserves_overlap_boundaries() {
    let c = config();
    for latent in 1..257 {
        let p = FastH3VideoDecodePlan::new(&c, latent, 1000).unwrap();
        // Independent length pattern of 17-frame encoder clips: first latent
        // contributes one frame, the next four contribute four apiece.
        let expected = 17 * ((latent - 1) / 5) + 4 * ((latent - 1) % 5) + 1;
        assert_eq!(p.output_frames(), expected);
        let mut published = 0;
        for i in 0..p.chunk_count() {
            let chunk = p.chunk(i).unwrap();
            assert_eq!(
                (chunk.latent_count, chunk.raw_frames, chunk.main_start),
                (7, 28, 3)
            );
            assert_eq!(
                (chunk.next_overlap_start, chunk.next_overlap_frames),
                (23, 5)
            );
            assert_eq!(chunk.output_start, published);
            assert_eq!(chunk.blend_prefix_frames, if i == 0 { 0 } else { 5 });
            for j in chunk.latent_start..chunk.latent_start + chunk.latent_count {
                assert_eq!(p.source_latent(j).unwrap(), j.min(latent - 1));
            }
            published += chunk.publish_frames;
        }
        assert_eq!(published + p.final_overlap_frames(), expected);
        assert!(p.final_overlap_frames() <= 5);
        assert!(p.chunk(p.chunk_count()).is_err());
        assert!(p.source_latent(latent + p.repeat_last_latents()).is_err());
    }
    assert_eq!(
        FastH3VideoDecodePlan::new(&c, 37, 124)
            .unwrap()
            .output_frames(),
        124
    );
    assert_eq!(
        FastH3VideoDecodePlan::new(&c, 72, 243)
            .unwrap()
            .output_frames(),
        243
    );
    assert!(FastH3VideoDecodePlan::new(&c, 37, 123).is_err());
    assert!(FastH3VideoDecodePlan::new(&c, 0, 1000).is_err());
    assert!(FastH3VideoDecodePlan::new(&c, usize::MAX, usize::MAX).is_err());
}

#[test]
fn config_rejects_incompatible_class_dimensions_and_statistics() {
    let c = config();
    assert_eq!(
        (
            c.hidden_size().unwrap(),
            c.ffn_size().unwrap(),
            c.rotary_dimension()
        ),
        (128, 256, 48)
    );
    assert_eq!(c.output_patch_width().unwrap(), 48);
    assert!(FastH3VideoVaeConfig::parse(&serde_json::to_vec(&config_json()).unwrap(), 1).is_err());
    for (key, value) in [
        ("_class_name", serde_json::json!("AutoencoderKLPH3Image")),
        ("temporal_downsample_factors", serde_json::json!([0, 2])),
        ("decoder_rope_dim_ratio", serde_json::json!(0.76)),
        ("latents_std", serde_json::json!([1.0, 0.0])),
        ("block_out_channels", serde_json::json!([31, 64])),
        ("token_drop", serde_json::json!(5)),
    ] {
        let mut j = config_json();
        j[key] = value;
        assert!(
            FastH3VideoVaeConfig::parse(&serde_json::to_vec(&j).unwrap(), 4096).is_err(),
            "{key}"
        );
    }
}

type Tensor = (Dtype, Vec<usize>, Vec<u8>);
struct Fixture {
    tensors: BTreeMap<String, Tensor>,
    metadata: serde_json::Value,
}
impl Fixture {
    fn new() -> Self {
        let mut f = Self {
            tensors: BTreeMap::new(),
            metadata: serde_json::json!({
                "block_size":16,"rotation_group":null,"skip_blocks":[],"act_scale":"unit",
                "num_linears":6,"student_layers":1
            }),
        };
        // Independent tiny export following original checkpoint names/dtypes.
        for (n, s) in [
            ("post_quant_conv.weight", vec![2, 2, 1, 1, 1]),
            ("post_quant_conv.bias", vec![2]),
            ("decoder.proj_in.weight", vec![128, 2]),
            ("decoder.proj_in.bias", vec![128]),
            ("decoder.register_tokens", vec![1, 4, 128]),
            ("decoder.norm_out.weight", vec![128]),
            ("decoder.norm_out.bias", vec![128]),
            ("decoder.proj_out.weight", vec![48, 128]),
            ("decoder.proj_out.bias", vec![48]),
        ] {
            f.put(n, s, Dtype::F32, 0);
        }
        for name in ["norm1.weight", "norm2.weight", "scale1", "scale2"] {
            f.put(
                &format!("decoder.transformer_blocks.0.{name}"),
                vec![128],
                Dtype::F32,
                0,
            );
        }
        for (suffix, n, k) in [
            ("attn.to_q", 128, 128),
            ("attn.to_k", 128, 128),
            ("attn.to_v", 128, 128),
            ("attn.to_out.0", 128, 128),
            ("ff.net.0.proj", 512, 128),
            ("ff.net.2", 128, 256),
        ] {
            let p = format!("decoder.transformer_blocks.0.{suffix}");
            f.put(
                &format!("{p}.packed_weight"),
                vec![n, k / 2],
                Dtype::U8,
                0x22,
            );
            f.put(
                &format!("{p}.weight_inv_scale"),
                vec![n, k / 16],
                Dtype::U8,
                0x38,
            );
            f.put(&format!("{p}.bias"), vec![n], Dtype::BF16, 0);
            f.put(&format!("{p}.input_amax"), vec![], Dtype::F32, 0);
            f.tensors.insert(
                format!("{p}.weight_global_sf"),
                (Dtype::F32, vec![], 3.3f32.to_le_bytes().to_vec()),
            );
        }
        f
    }
    fn put(&mut self, n: &str, s: Vec<usize>, d: Dtype, fill: u8) {
        let bytes = s.iter().product::<usize>() * d.size();
        self.tensors.insert(n.into(), (d, s, vec![fill; bytes]));
    }
    fn open(
        &self,
        l: FastH3VideoDecoderWeightLimits,
    ) -> (tempfile::TempDir, anyhow::Result<FastH3VideoDecoderPackage>) {
        let dir = tempfile::tempdir().unwrap();
        let metadata = Some(HashMap::from([
            (
                "format".into(),
                "fastvideo_h3_decoder_nvfp4_packed_v1".into(),
            ),
            (
                "nvfp4_metadata".into(),
                serde_json::to_string(&self.metadata).unwrap(),
            ),
        ]));
        let views = self
            .tensors
            .iter()
            .map(|(n, (d, s, b))| (n.as_str(), TensorView::new(*d, s.clone(), b).unwrap()));
        serialize_to_file(views, &metadata, &dir.path().join("decoder.safetensors")).unwrap();
        let a = SafetensorsArchive::open_single_file(dir.path(), "decoder.safetensors").unwrap();
        (dir, FastH3VideoDecoderPackage::new(&config(), a, l))
    }
    fn rejects(&self, needle: &str) {
        let (_dir, p) = self.open(limits());
        let e = p.err().expect("must reject");
        assert!(
            format!("{e:#}").contains(needle),
            "expected {needle}: {e:#}"
        );
    }
}

#[test]
fn original_packed_decoder_package_preserves_physical_identity_and_shared_owner() {
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    assert_eq!(p.components().len(), 43);
    assert_eq!(p.matrices().len(), 6);
    let mut bytes = 0;
    let mut retained = None;
    for c in p.components() {
        let v = p.component(c).unwrap();
        assert_eq!(v.source_file(), "decoder.safetensors");
        assert_eq!(v.bytes().len() as u64, c.physical_bytes().unwrap());
        assert!(v.retained_host_memory().is_some());
        retained = Some(v.retained_host_memory().unwrap().clone());
        bytes += v.bytes().len();
    }
    assert_eq!(p.source_bytes(), bytes);
    let o = p
        .nvfp4_weight_operands(
            "decoder.transformer_blocks.0.attn.to_q",
            WeightId::new("b").unwrap(),
            WeightId::new("sb").unwrap(),
        )
        .unwrap();
    let schema = WeightSchema {
        format_id: WeightFormatId::new("format.test.decoder").unwrap(),
        layout_id: WeightLayoutId::new("layout.test.decoder").unwrap(),
        version: ContractVersion::new(1, 0),
        tensors: vec![o.packed_values, o.scales],
        components: o.components.to_vec(),
    };
    schema
        .validate(&ModelFamilyId::new("family.test.decoder").unwrap())
        .unwrap();
    for (w, c) in schema.tensors.iter().zip(&schema.components) {
        let binding = ResolvedWeightBinding::from_schema(&schema, &w.id).unwrap();
        let physical = binding.physical_component_view().unwrap();
        assert_eq!(physical.encoding(), &c.encoding);
        assert_eq!(physical.component_id(), &c.id);
        assert!(w.logical_elements().is_err());
    }
    let mut altered = schema.components[0].clone();
    altered.role = WeightComponentRole::Values;
    assert!(p.component(&altered).is_err());
    drop(p);
    assert!(retained.is_some());
}

#[test]
fn packed_decoder_rejects_depth_cuts_dtypes_bad_scales_and_capacity() {
    for (k, v) in [
        ("student_layers", serde_json::json!(2)),
        ("skip_blocks", serde_json::json!([0])),
        ("act_scale", serde_json::json!("static")),
        ("num_linears", serde_json::json!(5)),
    ] {
        let mut f = Fixture::new();
        f.metadata[k] = v;
        f.rejects(if k == "student_layers" || k == "num_linears" {
            "every declared layer"
        } else {
            "recipe"
        });
    }
    let mut f = Fixture::new();
    f.tensors.remove("decoder.norm_out.bias");
    f.rejects("inventory");
    let mut f = Fixture::new();
    f.put("decoder.norm_out.bias", vec![128], Dtype::BF16, 0);
    f.rejects("physical declaration");
    let mut f = Fixture::new();
    f.tensors
        .get_mut("decoder.transformer_blocks.0.attn.to_q.weight_global_sf")
        .unwrap()
        .2 = 0f32.to_le_bytes().to_vec();
    f.rejects("positive");
    let mut f = Fixture::new();
    f.tensors
        .get_mut("decoder.transformer_blocks.0.attn.to_q.weight_inv_scale")
        .unwrap()
        .2[0] = 0x7f;
    f.rejects("scale");
    let f = Fixture::new();
    for l in [
        FastH3VideoDecoderWeightLimits {
            maximum_tensors: 42,
            ..limits()
        },
        FastH3VideoDecoderWeightLimits {
            maximum_source_bytes: 4,
            ..limits()
        },
        FastH3VideoDecoderWeightLimits {
            maximum_metadata_bytes: 4,
            ..limits()
        },
    ] {
        assert!(f.open(l).1.is_err());
    }
}

fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
#[test]
fn decoder_alpha_is_source_bound_f32_reciprocal_with_managed_retention() {
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    let recipe = p
        .alpha_derivation("decoder.transformer_blocks.0.attn.to_q")
        .unwrap();
    assert_eq!(recipe.stored_bits(), 3.3f32.to_bits());
    assert_eq!(recipe.result_bits().unwrap(), (1.0f32 / 3.3f32).to_bits());
    assert_ne!(recipe.result_bits().unwrap(), recipe.stored_bits()); // not the DiT recipe
    let account = host(4);
    let backing = recipe.managed_backing(&p, &account).unwrap();
    assert_eq!(backing.as_ref(), &(1.0f32 / 3.3f32).to_le_bytes());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 4);
    assert!(recipe.managed_backing(&p, &account).is_err());
    let mut wire = serde_json::to_value(&recipe).unwrap();
    wire["stored_bits"] = 2.0f32.to_bits().into();
    let altered: FastH3DecoderAlphaDerivation = serde_json::from_value(wire).unwrap();
    assert_ne!(
        altered.output_component().unwrap().id,
        recipe.output_component().unwrap().id
    );
    assert!(altered.managed_backing(&p, &host(4)).is_err());
    drop(p);
    assert_eq!(backing.as_ref(), &(1.0f32 / 3.3f32).to_le_bytes());
    drop(backing);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let (_dir, p) = Fixture::new().open(limits());
    let p = p.unwrap();
    let tiny = host(3);
    assert!(recipe.managed_backing(&p, &tiny).is_err());
    assert_eq!(tiny.usage().unwrap().allocated_bytes(), 0);
}
