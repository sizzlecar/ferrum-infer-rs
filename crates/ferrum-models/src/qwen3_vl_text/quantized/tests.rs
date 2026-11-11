use super::*;
use crate::qwen3_vl_text::{Qwen3VLTextArchitecture, Qwen3VLTextKeyRun, Qwen3VLTextLayout};
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use serde_json::json;

#[cfg(feature = "cuda")]
mod cuda;
mod oracle;
mod visual;

type Quantization = Qwen3VLTextComfyProjectionQuantization;
type Family = Qwen3VLTextComfyFamily;
type OriginalTensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
fn limits() -> Qwen3VLTextComfyLimits {
    Qwen3VLTextComfyLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 16 << 20,
        max_plan_nodes: 1024,
    }
}
fn id<T>(s: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn text(layers: u64, batch: usize) -> Qwen3VLTextConfig {
    let raw = json!({"architectures":["Qwen3VLForConditionalGeneration"], "model_type":"qwen3_vl", "dtype":"bfloat16",
        "text_config":{"model_type":"qwen3_vl_text", "dtype":"bfloat16", "attention_bias":false, "attention_dropout":0.0,
        "hidden_act":"silu", "num_hidden_layers":layers, "hidden_size":256, "intermediate_size":256,
        "num_attention_heads":2, "num_key_value_heads":1, "head_dim":128, "vocab_size":16,
        "max_position_embeddings":128, "rms_norm_eps":0.000001, "rope_theta":5000000,
        "rope_scaling":{"rope_type":"default", "mrope_interleaved":true, "mrope_section":[64,0,0]}}});
    let layout = Qwen3VLTextLayout::new(
        3,
        (0..batch)
            .map(|b| {
                if b == 0 {
                    vec![
                        Qwen3VLTextKeyRun::new(1, false),
                        Qwen3VLTextKeyRun::new(2, true),
                    ]
                } else {
                    vec![Qwen3VLTextKeyRun::new(3, true)]
                }
            })
            .collect(),
    )
    .unwrap();
    Qwen3VLTextConfig::new(
        Qwen3VLTextArchitecture::from_hf_value(&raw).unwrap(),
        layout,
    )
    .unwrap()
    .with_hidden_state_indices(if layers == 1 {
        vec![1]
    } else {
        vec![1, layers]
    })
    .unwrap()
}
fn config(text: Qwen3VLTextConfig, swapped: bool) -> Qwen3VLTextComfyConfig {
    let projections = weights::projection_shapes(&text)
        .unwrap()
        .into_iter()
        .enumerate()
        .map(|(i, (name, _, _))| {
            (
                name,
                if (i % 3 == 0) ^ swapped {
                    Quantization::W8A8
                } else {
                    Quantization::W4A8
                },
            )
        })
        .collect();
    Qwen3VLTextComfyConfig::new(text, projections).unwrap()
}
fn floats(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn original(c: &Qwen3VLTextComfyConfig) -> OriginalTensors {
    let mut tensors = BTreeMap::new();
    for (name, dimensions, dtype) in weights::entries(c).unwrap() {
        let shape: Vec<usize> = dimensions.iter().map(|&v| v as usize).collect();
        let count = shape.iter().product::<usize>();
        let (dtype, bytes) = if name.ends_with("norm.weight") {
            let values = (0..count).map(|i| 0.75 + ((i * 7) % 17) as f32 / 32.0);
            (
                Dtype::BF16,
                values
                    .flat_map(|v| bf16::from_f32(v).to_bits().to_le_bytes())
                    .collect(),
            )
        } else if name.ends_with("weight_s_rel") {
            (Dtype::F8_E4M3, vec![0x38; count])
        } else if name.ends_with("weight_s_channel") || name.ends_with("weight_scale") {
            // Small nonzero residual branches keep the independent scalar
            // oracle away from quantizer thresholds while exercising all ops.
            (
                Dtype::F32,
                floats((0..count).map(|i| (1 + i % 5) as f32 / 65536.0)),
            )
        } else if name.ends_with("weight_codebook") {
            (Dtype::F32, floats((0..16).map(|i| (i as f32 - 8.0) / 8.0)))
        } else {
            assert_eq!(dtype, ElementType::I8);
            let bytes = (0..count)
                .map(|i| {
                    let row = i / shape[1];
                    let column = i % shape[1];
                    if c.projections()[name.strip_suffix(".weight").unwrap()] == Quantization::W4A8
                    {
                        let lo = ((row * 3 + column * 7) % 16) as u8;
                        let hi = ((row * 5 + column * 11 + 1) % 16) as u8;
                        lo | (hi << 4)
                    } else {
                        (((row * 3 + column * 7) % 17) as i8 - 8) as u8
                    }
                })
                .collect();
            (Dtype::I8, bytes)
        };
        tensors.insert(name, (dtype, shape, bytes));
    }
    for (module, mode) in c.projections() {
        let bytes = serde_json::to_vec(&mode.metadata()).unwrap();
        tensors.insert(
            format!("{module}.comfy_quant"),
            (Dtype::U8, vec![bytes.len()], bytes),
        );
    }
    tensors
}
fn archive(tensors: &OriginalTensors) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let directory = tempfile::tempdir().unwrap();
    let views = tensors
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("weights.safetensors")).unwrap();
    let archive =
        SafetensorsArchive::open_single_file(directory.path(), "weights.safetensors").unwrap();
    (directory, ComfyCanonicalWeightSource::new(archive).unwrap())
}
fn inputs(c: &Qwen3VLTextComfyConfig) -> BTreeMap<ProgramValueId, Vec<f32>> {
    let t = c.text();
    let a = t.architecture();
    let rows = t.layout().rows() as usize;
    let h = a.hidden_size() as usize;
    let d = a.head_dim() as usize;
    let mut cosine = Vec::new();
    let mut sine = Vec::new();
    for row in 0..rows {
        for column in 0..d {
            let phase = (row as f32 + 0.25) * ((column % (d / 2) + 1) as f32) / 128.0;
            cosine.push(libm::cosf(phase));
            sine.push(libm::sinf(phase));
        }
    }
    BTreeMap::from([
        (
            id(Family::EMBEDDINGS_ID),
            (0..rows * h)
                .map(|i| {
                    let row = i / h;
                    let column = i % h;
                    ((column * 13 + row * 31) % 101) as f32 / 64.0 - 0.75
                        + row as f32 / 128.0
                        + 0.000123
                })
                .collect(),
        ),
        (id(Family::COSINE_ID), cosine),
        (id(Family::SINE_ID), sine),
    ])
}

#[test]
fn original_metadata_selects_each_projection_and_promotes_only_norm_vectors() {
    for swapped in [false, true] {
        let c = config(text(2, 2), swapped);
        let (_directory, source) = archive(&original(&c));
        let detected = Qwen3VLTextComfyConfig::from_checkpoint(c.text().clone(), &source).unwrap();
        assert_eq!(detected, c);
        let checkpoint = Qwen3VLTextComfyCheckpoint::new(&c, &source, limits()).unwrap();
        let family = Family::new(limits()).unwrap();
        let stored = family.weights(&c).unwrap();
        let schema = stored.stored_schema().unwrap();
        for component in &schema.components {
            let payload = checkpoint.component(component).unwrap();
            let name = &component.external_names[0];
            if name.ends_with("norm.weight") {
                assert_eq!(source.archive().tensor(name).unwrap().dtype(), Dtype::BF16);
                assert_eq!(payload.element_type(), ElementType::F32);
                let expected: Vec<u8> = source
                    .archive()
                    .tensor(name)
                    .unwrap()
                    .bytes()
                    .chunks_exact(2)
                    .flat_map(|b| {
                        bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap()))
                            .to_f32()
                            .to_le_bytes()
                    })
                    .collect();
                assert_eq!(payload.bytes(), expected);
            } else {
                assert!(payload.retained_host_memory().is_some());
            }
        }
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &id(Family::CUDA_PROFILE_ID),
            )
            .unwrap();
        let graph = prepared.program();
        assert!(prepared
            .numerical_profile()
            .boundaries
            .values()
            .all(|t| *t == ElementType::F32));
        assert!(graph
            .input_tensor_specs()
            .values()
            .all(|t| t.element_type == ElementType::F32));
        for (module, mode) in c.projections() {
            let weight_value = weights::value(&format!("{module}.weight")).unwrap();
            let nodes: Vec<_> = graph
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .filter(|n| n.inputs.contains(&weight_value))
                .collect();
            assert_eq!(nodes.len(), 1);
            assert_eq!(nodes[0].operation_id.as_str(), mode.operation());
        }
        let tap = Family::hidden_state_output_id(1).unwrap();
        assert!(graph.outputs().contains(&tap));
        let producer = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .find(|n| n.outputs.contains(&tap))
            .unwrap();
        assert_eq!(
            producer.operation_id.as_str(),
            PLAIN_RESIDUAL_ADD_F32_OPERATION_ID
        );
        let last = Family::hidden_state_output_id(2).unwrap();
        let norm = graph
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .find(|n| n.outputs.contains(&last))
            .unwrap();
        assert_eq!(norm.operation_id.as_str(), HEAD_RMS_NORM_F32_OPERATION_ID);
        assert_eq!(norm.inputs[0].as_str(), Family::OUTPUT_ID);
    }
}

#[test]
fn missing_or_mismatched_modules_visual_inputs_and_real_capacity_limits_fail_closed() {
    let c = config(text(1, 1), false);
    let mut missing = c.projections().clone();
    missing.pop_first();
    assert!(Qwen3VLTextComfyConfig::new(c.text().clone(), missing).is_err());
    let mut renamed = c.projections().clone();
    let (key, value) = renamed.pop_first().unwrap();
    renamed.insert(key.strip_prefix("model.").unwrap().into(), value);
    assert!(Qwen3VLTextComfyConfig::new(c.text().clone(), renamed).is_err());
    let mut bad = original(&c);
    let name = super::super::weights::external_name(0, "input_layernorm.weight");
    let norm = bad.get_mut(&name).unwrap();
    norm.0 = Dtype::F32;
    norm.2 = floats(std::iter::repeat_n(1.0, 256));
    let (_directory, source) = archive(&bad);
    assert!(Qwen3VLTextComfyCheckpoint::new(&c, &source, limits()).is_err());
    let other = config(c.text().clone(), true);
    let (_directory, source) = archive(&original(&other));
    assert!(Qwen3VLTextComfyCheckpoint::new(&c, &source, limits()).is_err());
    let mut low = limits();
    low.max_stored_bytes = 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
    let mut low = limits();
    low.max_plan_nodes = 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
    let visual = text(3, 1)
        .with_deepstack(
            super::super::Qwen3VLTextDeepStack::new(vec![super::super::Qwen3VLTextVisualRun::new(
                1, 1,
            )
            .unwrap()])
            .unwrap(),
        )
        .unwrap();
    assert!(Qwen3VLTextComfyConfig::new(visual, BTreeMap::new()).is_err());
    // Both source metadata formats remain explicit and independently checked.
    assert!(Quantization::from_metadata(
        &ferrum_quantization::comfy_convrot::ComfyQuantizationMetadata::Int8Tensorwise {
            convrot: false,
            convrot_groupsize: 256
        }
    )
    .is_err());
}
