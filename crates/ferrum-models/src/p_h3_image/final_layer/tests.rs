use super::*;
use ferrum_quantization::SafetensorsArchive;
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
type Payloads = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;
type Family = PH3ImageFinalLayerFamily;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn limits() -> PH3ImageFinalLayerLimits {
    PH3ImageFinalLayerLimits {
        max_metadata_bytes: 1 << 20,
        max_stored_bytes: 1 << 20,
        max_plan_nodes: 1024,
    }
}
fn transformer() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({
    "_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":8,"audio_in_channels":2,"ffn_dim":8,"final_norm_eps":0.75,
    "freq_dim":8,"hidden_size":4,"in_channels":2,"norm_eps":0.00001,"num_attention_heads":1,"num_layers":1,"num_refiner_layers":1,
    "patch_size":[1,1,2],"qk_norm_eps":0.00001,"rope_freq_dim":1,"rope_theta":100,"text_dim":4,"time_embed_dim":4,"time_embed_hidden_dim":8,
})).unwrap()).unwrap()
}
fn run(length: u64, time: u64) -> PH3ImageFinalLayerTimeRun {
    PH3ImageFinalLayerTimeRun::new(length, time).unwrap()
}
fn config(many: bool) -> PH3ImageFinalLayerConfig {
    let requests = if many {
        (0..33)
            .map(|i| {
                PH3ImageFinalLayerRequest::new(1, vec![run(4, i % 2)], vec![run(1, (i + 1) % 2)])
                    .unwrap()
            })
            .collect()
    } else {
        vec![
            PH3ImageFinalLayerRequest::new(
                3,
                vec![run(2, 0), run(2, 1)],
                vec![run(1, 1), run(2, 0)],
            )
            .unwrap(),
            PH3ImageFinalLayerRequest::new(2, vec![run(4, 1)], vec![run(2, 1)]).unwrap(),
        ]
    };
    PH3ImageFinalLayerConfig::new(
        &transformer(),
        PH3ImageCurveTimeEmbedding::new(3, 2).unwrap(),
        2,
        requests,
    )
    .unwrap()
}
fn f32_bytes(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_le_bytes).collect()
}
fn payloads(c: &PH3ImageFinalLayerConfig) -> Payloads {
    assert_eq!(c.hidden, 4);
    let mut p = Payloads::new();
    let affine_weight = [
        0., 0., 0.125, 0.25, -0.5, 0.125, 0.25, -0.25, 0., 0., 0.25, 0., 0., -0.25, 0.5, 0.25,
    ];
    let affine_bias = [
        -1.5,
        -0.25,
        0.1234567,
        0.,
        2.0_f32.powi(-23),
        0.1,
        0.5,
        -0.25,
    ];
    let video_weight = [
        16777216., 0., 0., 0., 0., 2., -0.5, 0.25, 0., -1., 1., 0., 0.5, 0.25, 0., -1.,
    ];
    let audio_weight = [-8388608., 0., 0., 0., 0., 0.5, -2., 1.];
    for (name, shape, dtype) in weights::entries(c) {
        let count = shape.iter().product::<u64>() as usize;
        let (dtype, bytes) = if dtype == ElementType::Bf16 {
            (
                Dtype::BF16,
                std::iter::repeat_n(bf16::from_f32(3.0).to_bits(), count)
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            )
        } else {
            let values: &[f32] = match name {
                "final_layer.adaln_proj.linear.weight" => &affine_weight,
                "final_layer.adaln_proj.linear.bias" => &affine_bias,
                "final_layer.video_out.weight" => &video_weight,
                "final_layer.video_out.bias" => &[0.5, -0.25, 0.125, -0.5],
                "final_layer.audio_out.weight" => &audio_weight,
                "final_layer.audio_out.bias" => &[0.25, -0.125],
                _ => unreachable!(),
            };
            assert_eq!(values.len(), count);
            (Dtype::F32, f32_bytes(values.iter().copied()))
        };
        p.insert(
            name.into(),
            (
                dtype,
                shape.into_iter().map(|v| v as usize).collect(),
                bytes,
            ),
        );
    }
    p.insert(
        "adaln_t_table".into(),
        (
            Dtype::F32,
            vec![3, 2],
            f32_bytes([0., 1., 0.5, 0.25, 1., -0.5]),
        ),
    );
    p
}
fn archive(p: &Payloads) -> (tempfile::TempDir, SafetensorsArchive) {
    let d = tempfile::tempdir().unwrap();
    let views = p
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &d.path().join("weights.safetensors")).unwrap();
    let a = SafetensorsArchive::open_single_file(d.path(), "weights.safetensors").unwrap();
    (d, a)
}
fn input(c: &PH3ImageFinalLayerConfig) -> Vec<u16> {
    c.requests
        .iter()
        .flat_map(|r| {
            std::iter::repeat_n(bf16::NAN.to_bits(), (r.prefix_rows() * c.hidden) as usize).chain(
                std::iter::repeat_n(
                    bf16::from_f32(0.5).to_bits(),
                    ((r.audio_rows().unwrap() + r.image_rows().unwrap()) * c.hidden) as usize,
                ),
            )
        })
        .collect()
}
fn f32_parameter(source: &SafetensorsArchive, name: &str) -> Vec<f32> {
    source
        .tensor(name)
        .unwrap()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn affine(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    bias.iter()
        .enumerate()
        .map(|(row, b)| {
            let mut sum = 0.0_f32;
            for i in 0..x.len() {
                sum += x[i] * weight[row * x.len() + i];
            }
            sum + *b
        })
        .collect()
}
fn expected(c: &PH3ImageFinalLayerConfig, source: &SafetensorsArchive, image: bool) -> Vec<u16> {
    let weight = f32_parameter(source, "final_layer.adaln_proj.linear.weight");
    let bias = f32_parameter(source, "final_layer.adaln_proj.linear.bias");
    let parameters = [
        affine(&[0.25, 0.625], &weight, &bias),
        affine(&[0.75, -0.125], &weight, &bias),
    ];
    let module = if image {
        "final_layer.video_out"
    } else {
        "final_layer.audio_out"
    };
    let w = f32_parameter(source, &format!("{module}.weight"));
    let b = f32_parameter(source, &format!("{module}.bias"));
    let mut output = Vec::new();
    for request in &c.requests {
        for run in if image {
            request.image_runs()
        } else {
            request.audio_runs()
        } {
            let p = &parameters[run.time_index() as usize];
            // x=.5, epsilon=.75, gamma=3 => RMS inverse exactly one, n=1.5.
            let modulated: Vec<_> = (0..4)
                .map(|i| {
                    let factor = 1.0_f32 + p[4 + i];
                    let product = 1.5_f32 * factor;
                    product + p[i]
                })
                .collect();
            assert_eq!(modulated[0], 2.0_f32.powi(-22));
            let head = affine(&modulated, &w, &b);
            for _ in 0..run.length() {
                output.extend(head.iter().map(|v| bf16::from_f32(*v).to_bits()));
            }
        }
    }
    output
}
#[test]
fn target_ranges_and_time_rows_are_explicit_and_conditioning_prefix_is_excluded() {
    let c = config(false);
    assert_eq!(
        (
            c.input_rows().unwrap(),
            c.image_rows().unwrap(),
            c.audio_rows().unwrap()
        ),
        (18, 5, 8)
    );
    let image = program::routes(&c, true).unwrap();
    let audio = program::routes(&c, false).unwrap();
    assert_eq!(
        image
            .iter()
            .map(|r| (r.source, r.length, r.time))
            .collect::<Vec<_>>(),
        [(7, 1, 1), (8, 2, 0), (16, 2, 1)]
    );
    assert_eq!(
        audio
            .iter()
            .map(|r| (r.source, r.length, r.time))
            .collect::<Vec<_>>(),
        [(3, 2, 0), (5, 2, 1), (12, 4, 1)]
    );
    assert!(PH3ImageFinalLayerRequest::new(0, vec![run(4, 0)], vec![run(1, 0)]).is_err());
    assert!(PH3ImageFinalLayerRequest::new(1, vec![run(3, 0)], vec![run(1, 0)]).is_err());
    let bad = PH3ImageFinalLayerRequest::new(1, vec![run(4, 2)], vec![run(1, 0)]).unwrap();
    assert!(PH3ImageFinalLayerConfig::new(
        &transformer(),
        PH3ImageCurveTimeEmbedding::new(3, 2).unwrap(),
        2,
        vec![bad]
    )
    .is_err());
}
#[test]
fn original_precision_single_head_schema_and_every_chunk_output_have_checked_geometry() {
    for many in [false, true] {
        let c = config(many);
        let family = Family::new(limits()).unwrap();
        let (_d, source) = archive(&payloads(&c));
        let checkpoint = PH3ImageFinalLayerCheckpoint::new(&c, &source, limits()).unwrap();
        let schema = family.weights(&c).unwrap();
        for component in &schema.stored_schema().unwrap().components {
            let p = checkpoint.component(component).unwrap();
            assert!(p.retained_host_memory().is_some());
            assert_eq!(
                p.bytes().as_ptr(),
                source
                    .tensor(&component.external_names[0])
                    .unwrap()
                    .bytes()
                    .as_ptr()
            );
        }
        let prepared = TypedFamilyRegistration::new(family)
            .prepare_with_profile(
                &serde_json::to_value(&c).unwrap(),
                &id(Family::REFERENCE_PROFILE_ID),
            )
            .unwrap();
        let options = c.compile_options().unwrap();
        for node in prepared.program().blocks().iter().flat_map(|b| &b.nodes) {
            for output in &node.outputs {
                assert!(options.tensor_specs().contains_key(output));
            }
            if node.operation_id.as_str() == ROW_SELECTED_MODULATION_BF16_F32_OPERATION_ID {
                assert_eq!(
                    options.tensor_specs()[&node.outputs[0]].element_type,
                    ElementType::F32
                );
            }
        }
        let schema = schema.stored_schema().unwrap();
        assert_eq!(schema.tensors.len(), weights::entries(&c).len());
        assert!(schema.components.iter().all(|w| w
            .external_names
            .iter()
            .all(|n| n.starts_with("final_layer."))));
    }
    let c = config(false);
    let mut p = payloads(&c);
    let item = p.get_mut("final_layer.video_out.weight").unwrap();
    item.1[0] *= 2;
    let more = item.2.clone();
    item.2.extend(more);
    let (_d, source) = archive(&p);
    assert!(PH3ImageFinalLayerCheckpoint::new(&c, &source, limits()).is_err());
    let mut low = limits();
    low.max_stored_bytes = 1;
    assert!(Family::new(low).unwrap().weights(&c).is_err());
}
mod execution;
