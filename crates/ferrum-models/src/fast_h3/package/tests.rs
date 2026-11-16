use std::collections::HashMap;

use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};

use super::*;
use crate::fast_h3::{FastH3Checkpoint, FastH3MetadataLimits};

type Tensor = (Dtype, Vec<usize>, Vec<u8>);

mod attention_core;
mod attention_input;
mod attention_output;
mod attention_projection;
mod dit_block;
mod ffn;
mod final_output;
mod input_projection;
mod modulation;
mod output_conditioning;
mod pdd_projection;
mod qk_norm_rope;
mod text_refiner;
mod time_conditioning;
mod vsa_pooling;

#[test]
fn package_exposes_original_nvfp4_components_as_checked_full_operand_views() {
    use ferrum_runtime::{
        ContractVersion, ModelFamilyId, ResolvedWeightBinding, WeightFormatId, WeightLayoutId,
        WeightSchema,
    };
    let (_dir, package) = Fixture::new().open(limits());
    let package = package.unwrap();
    let name = "transformer_blocks.0.attn.to_q.weight";
    let operands = package
        .nvfp4_weight_operands(
            name,
            WeightId::new("weight.b").unwrap(),
            WeightId::new("weight.sb").unwrap(),
        )
        .unwrap();
    assert_eq!(operands.geometry.columns(), 32);
    assert_eq!(operands.packed_values.dimensions, [48, 16]);
    assert_eq!(operands.scales.dimensions, [128, 4]);
    assert_eq!(operands.components[1].dimensions, [512]);
    let schema = WeightSchema {
        format_id: WeightFormatId::new("weight-format.safetensors.dense").unwrap(),
        layout_id: WeightLayoutId::new("weight-layout.test.fast-h3-selected").unwrap(),
        version: ContractVersion::new(1, 0),
        tensors: vec![operands.packed_values, operands.scales],
        components: operands.components.to_vec(),
    };
    schema
        .validate(&ModelFamilyId::new("family.test.fast-h3-selected").unwrap())
        .unwrap();
    for (tensor, component) in schema.tensors.iter().zip(&schema.components) {
        let binding = ResolvedWeightBinding::from_schema(&schema, &tensor.id).unwrap();
        let view = binding.physical_component_view().unwrap();
        assert_eq!(view.component_id(), &component.id);
        assert_eq!(view.encoding(), &component.encoding);
        let payload = package.component(component).unwrap();
        assert_eq!(payload.bytes().len() as u64, view.physical_bytes().unwrap());
        assert!(payload.retained_host_memory().is_some());
        assert!(tensor.logical_elements().is_err());
    }
    assert!(package
        .nvfp4_weight_operands(
            "transformer_blocks.1.attn.to_q.weight",
            WeightId::new("weight.b").unwrap(),
            WeightId::new("weight.sb").unwrap()
        )
        .is_err());
    assert!(package
        .nvfp4_weight_operands(
            name,
            WeightId::new("weight.b").unwrap(),
            WeightId::new("weight.b").unwrap()
        )
        .is_err());
}

fn config() -> FastH3TransformerConfig {
    config_with_attention(32, 3, 16, 48)
}
fn config_with_attention(
    hidden: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
) -> FastH3TransformerConfig {
    config_with_attention_and_rotary(hidden, heads, head_dim, ffn, 2)
}
fn config_with_attention_and_rotary(
    hidden: usize,
    heads: usize,
    head_dim: usize,
    ffn: usize,
    rope_freq_dim: usize,
) -> FastH3TransformerConfig {
    let mut transformer: serde_json::Value =
        serde_json::from_slice(include_bytes!("../fixtures/transformer.json")).unwrap();
    for (key, value) in [
        ("hidden_size", hidden),
        ("num_attention_heads", heads),
        ("attention_head_dim", head_dim),
        ("ffn_dim", ffn),
        ("freq_dim", 8),
        ("in_channels", 2),
        ("audio_in_channels", 4),
        ("num_layers", 2),
        ("num_refiner_layers", 1),
        ("text_dim", 16),
        ("time_embed_dim", 16),
        ("time_embed_hidden_dim", 32),
        ("rope_freq_dim", rope_freq_dim),
    ] {
        transformer[key] = value.into();
    }
    FastH3Checkpoint::parse(
        include_bytes!("../fixtures/inference.json"),
        &serde_json::to_vec(&transformer).unwrap(),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap()
    .transformer()
    .clone()
}

fn limits() -> FastH3WeightLimits {
    FastH3WeightLimits {
        maximum_tensors: 100,
        maximum_source_bytes: 1 << 20,
        maximum_skip_metadata_bytes: 1024,
    }
}

/// A tiny two-block export, built independently of the inventory function:
/// block zero is quantized, block one is retained dense, text refiner is dense.
/// Released diffusers aliases and the optional minimax_h3 packed prefix coexist.
struct Fixture {
    dense: BTreeMap<String, Tensor>,
    packed: BTreeMap<String, Tensor>,
    metadata: HashMap<String, String>,
}

impl Fixture {
    fn new() -> Self {
        Self::with_attention(&config())
    }
    fn with_attention(config: &FastH3TransformerConfig) -> Self {
        let h = config.hidden_size();
        let inner = config.attention_heads() * config.attention_head_dim();
        let ffn = config.ffn_dim();
        let mut f = Self {
            dense: BTreeMap::new(),
            packed: BTreeMap::new(),
            metadata: HashMap::from([("skip_blocks".into(), "[1]".into())]),
        };
        for (name, rows, cols) in [
            ("proj_in", h, 8),
            ("audio_proj_in", h, 4),
            ("time_embedder.linear_1", 32, 8),
            ("time_embedder.linear_2", 16, 32),
            ("proj_out", 256, h),
            ("audio_proj_out", 128, h),
            ("context_embedder", h, 16),
            ("norm_out.linear", 2 * h, 16),
        ] {
            f.linear(name, rows, cols, true);
        }
        f.dense("norm_out.norm.weight", &[h]);
        f.dense("token_refiner.final_norm.weight", &[h]);
        for prefix in [
            "transformer_blocks.0",
            "transformer_blocks.1",
            "token_refiner.refiner_blocks.0",
        ] {
            for name in ["norm1.weight", "norm2.weight"] {
                f.dense(&format!("{prefix}.{name}"), &[h]);
            }
            for name in ["attn.norm_q.weight", "attn.norm_k.weight"] {
                f.dense(&format!("{prefix}.{name}"), &[config.attention_head_dim()]);
            }
            for (name, rows, cols) in [
                ("attn.to_q", inner, h),
                ("attn.to_k", inner, h),
                ("attn.to_v", inner, h),
                ("attn.to_out.0", h, inner),
                ("ff.net.0.proj", 2 * ffn, h),
                ("ff.net.2", h, ffn),
            ] {
                if prefix == "transformer_blocks.0" {
                    let canonical = name
                        .replace("to_out.0", "to_out")
                        .replace("net.0.proj", "fc_in")
                        .replace("net.2", "fc_out");
                    f.quantized(
                        &format!("{prefix}.{canonical}"),
                        rows,
                        cols,
                        name.starts_with("ff."),
                    );
                } else {
                    f.linear(&format!("{prefix}.{name}"), rows, cols, false);
                }
            }
            if prefix.starts_with("transformer_blocks") {
                f.linear(&format!("{prefix}.adaln_proj.linear"), 18 * h, 16, true);
                let gate = format!("{prefix}.attn.to_gate_compress");
                if prefix == "transformer_blocks.0" {
                    f.quantized(&gate, inner, h, false);
                } else {
                    f.linear(&gate, inner, h, false);
                }
            }
        }
        f
    }
    fn dense(&mut self, name: &str, shape: &[usize]) {
        self.dense.insert(
            name.to_owned(),
            (
                Dtype::BF16,
                shape.to_vec(),
                vec![0; shape.iter().product::<usize>() * 2],
            ),
        );
    }
    fn linear(&mut self, name: &str, rows: usize, cols: usize, bias: bool) {
        self.dense(&format!("{name}.weight"), &[rows, cols]);
        if bias {
            self.dense(&format!("{name}.bias"), &[rows]);
        }
    }
    fn quantized(&mut self, name: &str, rows: usize, cols: usize, static_input: bool) {
        let prefix = format!("minimax_h3.{name}");
        let geometry = Nvfp4Geometry::new(rows, cols).unwrap();
        for (suffix, tensor) in [
            (
                "_nvfp4_weight",
                (Dtype::U8, vec![rows, cols / 2], vec![0x21; rows * cols / 2]),
            ),
            (
                "_nvfp4_weight_scale",
                (
                    Dtype::U8,
                    vec![geometry.scale_bytes()],
                    vec![0x38; geometry.scale_bytes()],
                ),
            ),
            (
                "_weight_global_sf",
                (
                    Dtype::BF16,
                    vec![],
                    half::bf16::from_f32(3.328125)
                        .to_bits()
                        .to_le_bytes()
                        .to_vec(),
                ),
            ),
            (
                "_nvfp4_alpha",
                (Dtype::F32, vec![], 0.3f32.to_le_bytes().to_vec()),
            ),
        ] {
            self.packed.insert(format!("{prefix}::{suffix}"), tensor);
        }
        if static_input {
            self.packed.insert(
                format!("{prefix}::_nvfp4_input_global_sf"),
                (Dtype::F32, vec![], 8.0f32.to_le_bytes().to_vec()),
            );
        }
    }
    fn open(
        &self,
        limits: FastH3WeightLimits,
    ) -> (tempfile::TempDir, Result<FastH3TransformerPackage>) {
        self.open_with_config(&config(), limits)
    }
    fn open_with_config(
        &self,
        config: &FastH3TransformerConfig,
        limits: FastH3WeightLimits,
    ) -> (tempfile::TempDir, Result<FastH3TransformerPackage>) {
        let dir = tempfile::tempdir().unwrap();
        for (name, tensors, metadata) in [
            ("dense.safetensors", &self.dense, None),
            (
                "nvfp4_weights.safetensors",
                &self.packed,
                Some(self.metadata.clone()),
            ),
        ] {
            let views = tensors.iter().map(|(name, (dtype, shape, bytes))| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            });
            serialize_to_file(views, &metadata, &dir.path().join(name)).unwrap();
        }
        let dense = SafetensorsArchive::open_single_file(dir.path(), "dense.safetensors").unwrap();
        let packed =
            SafetensorsArchive::open_single_file(dir.path(), "nvfp4_weights.safetensors").unwrap();
        (
            dir,
            FastH3TransformerPackage::new(
                config,
                dense,
                packed,
                FastH3Nvfp4Profile::DiTWithVsa,
                limits,
            ),
        )
    }
    fn reject(&self, expected: &str) {
        let (_dir, result) = self.open(limits());
        let error = match result {
            Ok(_) => panic!("accepted invalid fixture: {expected}"),
            Err(e) => e,
        };
        assert!(
            format!("{error:#}").contains(expected),
            "expected {expected:?}, got {error:#}"
        );
    }
}

#[test]
fn transformer_package_covers_dense_and_packed_parameters_without_dtype_or_alpha_loss() {
    let mut fixture = Fixture::new();
    // Some exports retain the original dense matrix; it is checked, excluded
    // from executable components and never used as a quantized fallback.
    fixture.dense("transformer_blocks.0.attn.to_q.weight", &[48, 32]);
    let (_dir, package) = fixture.open(limits());
    let package = package.unwrap();
    assert_eq!(package.weights().len(), 54);
    assert_eq!(package.components().len(), 77);
    assert_eq!(package.skip_blocks(), &[1]);
    assert_eq!(
        package.retained_dense_originals(),
        &["transformer_blocks.0.attn.to_q.weight"]
    );
    let q = package
        .nvfp4_matrix("transformer_blocks.0.attn.to_q.weight")
        .unwrap();
    assert_eq!(q.geometry().rows(), 48);
    assert_eq!(q.geometry().columns(), 32);
    assert_eq!(q.output_multiplier(), 0.3);
    assert_ne!(q.output_multiplier(), 1.0f32 / 3.328125);
    assert_eq!(q.value(0, 0).unwrap(), 0.15);
    assert_eq!(q.input_global_scale(), None);
    assert_eq!(
        package
            .nvfp4_matrix("transformer_blocks.0.ff.fc_in.weight")
            .unwrap()
            .input_global_scale(),
        Some(8.0)
    );
    assert!(package
        .nvfp4_matrix("transformer_blocks.1.attn.to_q.weight")
        .is_err());
    let head = package.weight("proj_out.weight").unwrap();
    assert_eq!(head.dimensions, [256, 32]);
    assert_eq!(head.compute_element_type, ElementType::F32);
    let FastH3WeightStorage::Dense(head) = &head.storage else {
        panic!("PDD head must stay dense")
    };
    assert_eq!(head.dense_element_type(), Some(ElementType::Bf16));
    let mut bytes = 0usize;
    for c in package.components() {
        let payload = package.component(c).unwrap();
        assert!(payload.retained_host_memory().is_some());
        assert_eq!(payload.bytes().len() as u64, c.physical_bytes().unwrap());
        bytes += payload.bytes().len();
    }
    assert_eq!(package.source_bytes(), bytes + 48 * 32 * 2);
    let mut tampered = head.clone();
    tampered.encoding = WeightEncoding::Dense {
        element_type: ElementType::F32,
    };
    assert!(package.component(&tampered).is_err());
}

#[test]
fn transformer_package_rejects_missing_duplicate_and_unexpected_coverage() {
    let mut f = Fixture::new();
    f.dense.remove("proj_out.weight");
    f.reject("missing dense FastH3 parameter proj_out.weight");
    let mut f = Fixture::new();
    f.packed
        .remove("minimax_h3.transformer_blocks.0.attn.to_q::_nvfp4_weight");
    f.dense("transformer_blocks.0.attn.to_q.weight", &[48, 32]);
    f.reject("missing or ambiguous NVFP4 matrix");
    let mut f = Fixture::new();
    f.dense("transformer_blocks.1.ff.fc_in.weight", &[96, 32]);
    f.reject("ambiguous dense aliases");
    let mut f = Fixture::new();
    f.quantized("transformer_blocks.1.attn.to_q", 48, 32, false);
    f.reject("unrecognized FastH3 NVFP4 parameters");
    let mut f = Fixture::new();
    f.dense("unrecognized.bias", &[1]);
    f.reject("unrecognized FastH3 dense parameters");
    let mut f = Fixture::new();
    f.packed.insert(
        "minimax_h3.transformer_blocks.0.attn.to_q::unknown_buffer".into(),
        (Dtype::U8, vec![1], vec![0]),
    );
    f.reject("unrecognized FastH3 NVFP4 parameters");
}

#[test]
fn transformer_package_checks_shapes_and_quantized_sidecars() {
    let mut f = Fixture::new();
    f.dense("proj_out.weight", &[8, 32]); // Base single head is not a widened PDD head.
    f.reject("dense shape differs for proj_out.weight");
    let mut f = Fixture::new();
    let tensor = f
        .packed
        .get_mut("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_weight")
        .unwrap();
    tensor.1 = vec![32, 48]; // Same byte count, wrong matrix geometry.
    f.reject("packed");
    let mut f = Fixture::new();
    f.packed
        .get_mut("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_alpha")
        .unwrap()
        .2 = f32::NAN.to_le_bytes().to_vec();
    f.reject("positive");
    let mut f = Fixture::new();
    f.packed
        .get_mut("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_input_global_sf")
        .unwrap()
        .2 = 0f32.to_le_bytes().to_vec();
    f.reject("positive");
    let mut f = Fixture::new();
    f.packed
        .remove("minimax_h3.transformer_blocks.0.ff.fc_in::_nvfp4_alpha");
    f.reject("absent");
    let mut f = Fixture::new();
    f.dense
        .insert("proj_in.bias".into(), (Dtype::U8, vec![32], vec![0; 32]));
    f.reject("floating-point storage");
}

#[test]
fn transformer_package_enforces_skip_metadata_and_source_capacity() {
    for raw in ["[true]", "[-1]", "[2]", "{}", "[0]"] {
        let mut f = Fixture::new();
        f.metadata.insert("skip_blocks".into(), raw.into());
        assert!(f.open(limits()).1.is_err(), "{raw}");
    }
    let f = Fixture::new();
    for l in [
        FastH3WeightLimits {
            maximum_tensors: 76,
            ..limits()
        },
        FastH3WeightLimits {
            maximum_source_bytes: 100,
            ..limits()
        },
        FastH3WeightLimits {
            maximum_skip_metadata_bytes: 1,
            ..limits()
        },
    ] {
        assert!(f.open(l).1.is_err());
    }
    let mut f = Fixture::new();
    f.metadata.insert("skip_blocks".into(), "[1, 1]".into());
    assert_eq!(f.open(limits()).1.unwrap().skip_blocks(), &[1]);
}

#[test]
fn packed_export_without_skip_metadata_requires_every_main_block() {
    let mut f = Fixture::new();
    f.metadata.clear();
    for (external, canonical, rows, cols, static_input) in [
        ("attn.to_q", "attn.to_q", 48, 32, false),
        ("attn.to_k", "attn.to_k", 48, 32, false),
        ("attn.to_v", "attn.to_v", 48, 32, false),
        ("attn.to_out.0", "attn.to_out", 32, 48, false),
        (
            "attn.to_gate_compress",
            "attn.to_gate_compress",
            48,
            32,
            false,
        ),
        ("ff.net.0.proj", "ff.fc_in", 96, 32, true),
        ("ff.net.2", "ff.fc_out", 32, 48, true),
    ] {
        f.dense
            .remove(&format!("transformer_blocks.1.{external}.weight"));
        f.quantized(
            &format!("transformer_blocks.1.{canonical}"),
            rows,
            cols,
            static_input,
        );
    }
    // Actual released export uses unqualified names and scalar-rank sidecars.
    f.packed = f
        .packed
        .into_iter()
        .map(|(name, t)| (name.strip_prefix("minimax_h3.").unwrap().to_owned(), t))
        .collect();
    let (_dir, package) = f.open(limits());
    let package = package.unwrap();
    assert!(package.skip_blocks().is_empty());
    assert!(package.retained_dense_originals().is_empty());
    assert_eq!(package.components().len(), 100);
    assert!(package
        .nvfp4_matrix("transformer_blocks.1.attn.to_gate_compress.weight")
        .is_ok());
}

#[test]
fn configured_full_model_inventory_has_real_non_square_projection_dimensions() {
    let checkpoint = FastH3Checkpoint::parse(
        include_bytes!("../fixtures/inference.json"),
        include_bytes!("../fixtures/transformer.json"),
        include_bytes!("../fixtures/video_scheduler.json"),
        include_bytes!("../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 32,
        },
    )
    .unwrap();
    let entries = inventory::entries(
        checkpoint.transformer(),
        FastH3Nvfp4Profile::DiTWithVsa,
        &[],
        1000,
    )
    .unwrap();
    let entry = |name: &str| entries.iter().find(|e| e.name == name).unwrap();
    assert_eq!(
        entry("transformer_blocks.0.attn.to_q.weight").shape,
        [7168, 5376]
    );
    assert_eq!(
        entry("transformer_blocks.0.attn.to_out.weight").shape,
        [5376, 7168]
    );
    assert_eq!(
        entry("transformer_blocks.0.ff.fc_in.weight").shape,
        [28672, 5376]
    );
    assert_eq!(
        entry("transformer_blocks.0.adaln_proj.linear.weight").shape,
        [96768, 2688]
    );
    assert_eq!(entry("proj_out.weight").shape, [3072, 5376]);
    assert_eq!(entry("audio_proj_out.weight").shape, [1024, 5376]);
    assert!(entry("transformer_blocks.41.attn.to_gate_compress.weight").packed);
    assert!(entry("transformer_blocks.49.attn.to_gate_compress.weight").packed);
    assert!(!entry("token_refiner.refiner_blocks.0.attn.to_q.weight").packed);
    assert_eq!(entries.len(), 688);
    assert_eq!(entries.iter().filter(|e| e.packed).count(), 350);
}
