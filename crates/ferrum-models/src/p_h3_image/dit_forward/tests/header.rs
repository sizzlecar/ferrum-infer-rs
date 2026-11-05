//! Original header contract only. This test loads no model payload, executes
//! no CUDA and cannot establish image quality or runtime capacity.
use super::super::super::PH3ImageProjectionStorage;
use super::*;
use ferrum_quantization::comfy_convrot::ComfyQuantizationMetadata;
#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<u64>,
}
#[derive(Deserialize)]
struct Blocks {
    tensors: BTreeMap<String, Tensor>,
    projection_metadata: BTreeMap<String, ComfyQuantizationMetadata>,
}
#[test]
fn complete_original_fifty_layer_dit_schema_matches_all_1074_physical_headers() {
    let blocks: Blocks = serde_json::from_str(include_str!(
        "../../curve_stack/tests/comfy_dit_block_headers.json"
    ))
    .unwrap();
    let mut header: BTreeMap<String, Tensor> =
        serde_json::from_str(include_str!("comfy_dit_nonblock_headers.json")).unwrap();
    header.extend(blocks.tensors);
    let t=PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({"_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":128,"audio_in_channels":32,"ffn_dim":14336,"final_norm_eps":1e-5,"freq_dim":256,"hidden_size":5376,"in_channels":24,"norm_eps":1e-5,"num_attention_heads":56,"num_layers":50,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.,"text_dim":5120,"time_embed_dim":2688,"time_embed_hidden_dim":5376})).unwrap()).unwrap();
    let storage = (0..t.layers())
        .map(|layer| {
            let get = |name: &str| {
                PH3ImageProjectionStorage::from_metadata(
                    &blocks.projection_metadata[&format!("blocks.{layer}.{name}")],
                )
                .unwrap()
            };
            PH3ImageBlockProjectionStorage::new(
                get("attn.qkv_proj"),
                get("attn.out_proj"),
                get("mlp.fc1"),
                get("mlp.fc2"),
            )
        })
        .collect();
    let tags = [
        PH3ImageModality::Image,
        PH3ImageModality::Text,
        PH3ImageModality::Text,
    ];
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = Some(PH3ImageReferenceImage::new(2, 4).unwrap());
    let layout = PH3ImageRequestLayout::new(&t, &tags, target, reference.as_slice()).unwrap();
    let times = vec![0; layout.rows()];
    let request = PH3ImageDiTRequest::new(&t, &tags, target, reference, &times, 1 << 20).unwrap();
    let c = PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        1,
        vec![request],
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::PerLayer(storage),
    )
    .unwrap();
    let limits = PH3ImageDiTLimits {
        max_metadata_bytes: 128 << 20,
        max_plan_nodes: 4096,
        max_stored_bytes: 64 << 30,
    };
    let composed = program::compose(&c, limits).unwrap();
    let mut actual = BTreeSet::new();
    for component in &composed.schema.components {
        let name = &component.external_names[0];
        let tensor = &header[name];
        assert_eq!(component.dimensions, tensor.shape, "original shape {name}");
        let dtype = if name.ends_with(".weight_s_rel") {
            assert_eq!(component.physical_element_type(), ElementType::U8);
            "F8_E4M3"
        } else {
            match component.physical_element_type() {
                ElementType::Bf16 => "BF16",
                ElementType::F32 => "F32",
                ElementType::I8 => "I8",
                other => panic!("unexpected physical type {other:?}"),
            }
        };
        assert_eq!(dtype, tensor.dtype, "original dtype {name}");
        assert!(actual.insert(name.clone()));
    }
    let expected = header
        .keys()
        .filter(|name| {
            !name.ends_with(".comfy_quant")
                && !matches!(name.as_str(), "adaln_t_table" | "rope.inv_freq")
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 1074);
    assert_eq!(header["adaln_t_table"].dtype, "F32");
    assert_eq!(header["adaln_t_table"].shape, [4097, 16]);
    assert_eq!(header["rope.inv_freq"].dtype, "F32");
    assert_eq!(header["rope.inv_freq"].shape, [16]);
}
