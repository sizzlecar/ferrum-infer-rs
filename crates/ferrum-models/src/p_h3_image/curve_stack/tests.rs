use super::super::{
    PH3ImageModality, PH3ImagePackedBatchLayout, PH3ImageReferenceImage, PH3ImageRequestLayout,
};
use super::*;
use ferrum_quantization::{comfy_convrot::ComfyCanonicalWeightSource, SafetensorsArchive};
use half::bf16;
fn transformer(layers: u64, original_width: bool) -> PH3ImageTransformerConfig {
    let (h, heads, ffn) = if original_width {
        (5376, 56, 14336)
    } else {
        (256, 4, 512)
    };
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({"_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":128,"audio_in_channels":32,"ffn_dim":ffn,"final_norm_eps":1e-5,"freq_dim":256,"hidden_size":h,"in_channels":24,"norm_eps":1e-5,"num_attention_heads":heads,"num_layers":layers,"num_refiner_layers":2,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":16,"rope_theta":10000.,"text_dim":5120,"time_embed_dim":2688,"time_embed_hidden_dim":5376})).unwrap()).unwrap()
}
fn limits() -> PH3ImageCurveStackLimits {
    PH3ImageCurveStackLimits {
        max_metadata_bytes: 16 << 20,
        max_plan_nodes: 2000,
        max_stored_bytes: 16 << 30,
    }
}
struct Scene {
    transformer: PH3ImageTransformerConfig,
    config: PH3ImageCurveStackConfig,
    positions: Vec<[f64; 3]>,
    x: Vec<u16>,
    times: Vec<f32>,
}
fn scene(layers: u64, original: bool, gpu_size: bool) -> Scene {
    let t = transformer(layers, original);
    let refs = [PH3ImageReferenceImage::new(4, 4).unwrap()];
    let target = if gpu_size {
        PH3ImageReferenceImage::new(20, 100).unwrap()
    } else {
        PH3ImageReferenceImage::new(4, 4).unwrap()
    };
    let tags0 = [
        PH3ImageModality::Text,
        PH3ImageModality::Image,
        PH3ImageModality::Text,
        PH3ImageModality::Text,
    ];
    let tags1 = [PH3ImageModality::Text; 8];
    let layouts = [
        PH3ImageRequestLayout::new(&t, &tags0, target, &refs).unwrap(),
        PH3ImageRequestLayout::new(&t, &tags1, target, &refs).unwrap(),
    ];
    let requests = layouts
        .iter()
        .enumerate()
        .map(|(request, layout)| {
            let mut times = vec![request as u64; layout.rows()];
            times[layout.reference_rows()].fill(2);
            PH3ImageCurveBlockRequest::from_layout(*layout, &times, limits().max_metadata_bytes)
                .unwrap()
        })
        .collect();
    let config = PH3ImageCurveStackConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        requests,
    )
    .unwrap();
    let packed = PH3ImagePackedBatchLayout::new(&layouts).unwrap();
    let mut positions = vec![[0.; 3]; packed.rows()];
    packed.write_positions(&mut positions).unwrap();
    let x = if original {
        Vec::new()
    } else {
        (0..packed.rows())
            .flat_map(|row| {
                let (request, local) = packed.request_at_row(row).unwrap();
                let tag = layouts[request].modality(local).unwrap() as u32;
                (0..t.hidden_size() as usize).map(move |column| {
                    bf16::from_f32(
                        (((row % 7) * 13 + column * 5) % 31) as f32 / 32. - 0.46875
                            + tag as f32 / 16.
                            + request as f32 / 8.,
                    )
                    .to_bits()
                })
            })
            .collect()
    };
    Scene {
        transformer: t,
        config,
        positions,
        x,
        times: vec![0.125, 0.625, 0.999],
    }
}
#[test]
fn two_layers_reuse_actual_block_contracts_weights_and_all_transport_shapes() {
    let s = scene(2, false, false);
    let b = compose::build(&s.config, limits()).unwrap();
    let templates = compose::admit(&s.config, limits()).unwrap();
    let source = templates.layer(&s.config, 0).unwrap();
    assert_eq!(b.schema.components.len(), 44);
    assert_eq!(b.schema.tensors.len(), 44);
    assert_eq!(
        b.program.blocks()[0].nodes.len(),
        source.program.blocks()[0].nodes.len() * s.config.layers() as usize
    );
    assert_eq!(
        b.program.outputs(),
        &[ProgramValueId::new(PH3ImageCurveStackFamily::OUTPUT_ID).unwrap()]
    );
    assert!(b.options.completion_retention().is_empty());
    let expected = b
        .program
        .inputs()
        .iter()
        .chain(
            b.program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs),
        )
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(expected, b.options.tensor_specs().keys().cloned().collect());
    b.schema
        .validate_program_references(&b.profile.family_id, &b.program)
        .unwrap();
    b.profile.validate_program(&b.program).unwrap();
    assert_eq!(b.profile.operations, source.profile.operations);
    for layer in 0..2 {
        for original in &source.schema.components {
            let id = remap::weight(layer, &original.id).unwrap();
            let mapped = b.schema.components.iter().find(|v| v.id == id).unwrap();
            assert_eq!(mapped.dimensions, original.dimensions);
            assert_eq!(mapped.encoding, original.encoding);
            assert_eq!(
                mapped.external_names,
                vec![remap::external(layer, &original.external_names[0]).unwrap()]
            );
        }
        for original in source.program.blocks().iter().flat_map(|b| &b.nodes) {
            let mapped = b.program.blocks()[0]
                .nodes
                .iter()
                .find(|n| n.id == remap::node(layer, &original.id).unwrap())
                .unwrap();
            assert_eq!(mapped.operation_id, original.operation_id);
            assert_eq!(mapped.attributes, original.attributes);
            assert_eq!(mapped.required_version, original.required_version);
            for (source_input, mapped_input) in original.inputs.iter().zip(&mapped.inputs) {
                if source_input.as_str() == PH3ImageCurveBlockFamily::INPUT_ID {
                    assert_eq!(
                        mapped_input,
                        &if layer == 0 {
                            ProgramValueId::new(PH3ImageCurveStackFamily::INPUT_ID).unwrap()
                        } else {
                            remap::output(layer - 1, 2).unwrap()
                        }
                    );
                }
            }
        }
        for (&input, expected) in [
            (
                &PH3ImageCurveBlockFamily::BASIS_ID,
                PH3ImageCurveStackFamily::BASIS_ID,
            ),
            (
                &PH3ImageCurveBlockFamily::COSINE_ID,
                PH3ImageCurveStackFamily::COSINE_ID,
            ),
            (
                &PH3ImageCurveBlockFamily::SINE_ID,
                PH3ImageCurveStackFamily::SINE_ID,
            ),
        ] {
            let n = b.program.blocks()[0]
                .nodes
                .iter()
                .filter(|n| n.id.as_str().contains(&format!(".layer{layer}.")))
                .flat_map(|n| &n.inputs)
                .filter(|v| v.as_str() == expected)
                .count();
            assert!(n > 0, "shared input {input}");
        }
    }
    let actual = serde_json::to_vec(&(&b.schema, &b.program, &b.profile, b.options.tensor_specs()))
        .unwrap()
        .len();
    assert!(actual <= compose::metadata_bound(&s.config, &templates).unwrap());
    TypedFamilyRegistration::new(PH3ImageCurveStackFamily::new(limits()).unwrap())
        .prepare_with_profile(
            &serde_json::to_value(&s.config).unwrap(),
            &NumericalProfileId::new(PH3ImageCurveStackFamily::CUDA_PROFILE_ID).unwrap(),
        )
        .unwrap();
}
#[test]
fn original_fifty_layer_geometry_is_admitted_only_with_declared_complete_capacities() {
    let s = scene(50, true, false);
    assert_eq!(s.config.layers(), 50);
    let t = compose::admit(&s.config, limits()).unwrap();
    let base = t.layer(&s.config, 0).unwrap();
    let nodes = base
        .program
        .blocks()
        .iter()
        .map(|b| b.nodes.len())
        .sum::<usize>()
        * 50;
    let stored = base
        .schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum::<u64>()
        * 50;
    let metadata = compose::metadata_bound(&s.config, &t).unwrap();
    let exact = PH3ImageCurveStackLimits {
        max_metadata_bytes: metadata,
        max_plan_nodes: nodes,
        max_stored_bytes: stored,
    };
    let b = compose::build(&s.config, exact).unwrap();
    assert_eq!(b.program.blocks()[0].nodes.len(), nodes);
    assert_eq!(b.schema.components.len(), 1100);
    assert_eq!(b.program.outputs().len(), 1);
    for denied in [
        PH3ImageCurveStackLimits {
            max_metadata_bytes: metadata - 1,
            ..exact
        },
        PH3ImageCurveStackLimits {
            max_plan_nodes: nodes - 1,
            ..exact
        },
        PH3ImageCurveStackLimits {
            max_stored_bytes: stored - 1,
            ..exact
        },
    ] {
        assert!(compose::admit(&s.config, denied).is_err());
    }
    let mut bad = serde_json::to_value(&s.config).unwrap();
    bad["layers"] = serde_json::json!(0);
    assert!(serde_json::from_value::<PH3ImageCurveStackConfig>(bad).is_err());
}
/// Minimized original Comfy W4 checkpoint header (dtype/shape only; offsets
/// omitted) and its two recorded canonical quant metadata schemas. No test
/// branches on an artifact path, revision identity, or special layer number.
#[test]
fn real_fifty_layer_header_and_quant_metadata_select_exact_mixed_physical_storage() {
    #[derive(Deserialize)]
    struct Tensor {
        dtype: String,
        shape: Vec<u64>,
    }
    #[derive(Deserialize)]
    struct Header {
        tensors: BTreeMap<String, Tensor>,
        projection_metadata:
            BTreeMap<String, ferrum_quantization::comfy_convrot::ComfyQuantizationMetadata>,
    }
    let header: Header =
        serde_json::from_str(include_str!("tests/comfy_dit_block_headers.json")).unwrap();
    let mut s = scene(50, true, false);
    let mut storage = Vec::new();
    for layer in 0..s.config.layers() {
        let get = |name: &str| {
            PH3ImageProjectionStorage::from_metadata(
                &header.projection_metadata[&format!("blocks.{layer}.{name}")],
            )
            .unwrap()
        };
        storage.push(PH3ImageBlockProjectionStorage::new(
            get("attn.qkv_proj"),
            get("attn.out_proj"),
            get("mlp.fc1"),
            get("mlp.fc2"),
        ));
    }
    s.config = s
        .config
        .with_projection_storage(PH3ImageStackProjectionStorage::PerLayer(storage))
        .unwrap();
    let composed = compose::build(&s.config, limits()).unwrap();
    let mut actual = BTreeSet::new();
    let mut w4 = 0usize;
    let mut w8 = 0usize;
    for component in &composed.schema.components {
        let name = &component.external_names[0];
        let source = &header.tensors[name];
        assert_eq!(component.dimensions, source.shape, "original shape {name}");
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
        assert_eq!(dtype, source.dtype, "original dtype {name}");
        actual.insert(name.clone());
    }
    let expected = header
        .tensors
        .keys()
        .filter(|name| !name.ends_with(".comfy_quant"))
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 1036);
    for layer in 0..s.config.layers() {
        for (_, storage) in s.config.storage_for_layer(layer).unwrap().modules() {
            match storage {
                PH3ImageProjectionStorage::CodebookW4A8 => w4 += 1,
                PH3ImageProjectionStorage::Int8W8A8 => w8 += 1,
            }
        }
    }
    assert_eq!((w4, w8), (168, 32));
    let ops = composed
        .profile
        .operations
        .iter()
        .map(|o| o.operation_id.as_str())
        .collect::<BTreeSet<_>>();
    assert!(
        ops.contains(COMFY_CONVROT_W4A8_BF16_OPERATION_ID)
            && ops.contains(COMFY_CONVROT_W8A8_BF16_OPERATION_ID)
            && ops.contains(COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID)
            && ops.contains(COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID)
    );
    assert_eq!(
        composed.profile.id.as_str(),
        PH3ImageCurveStackFamily::CUDA_MIXED_PROFILE_ID
    );
    assert_eq!(composed.program.outputs().len(), 1);
}
type Payloads = BTreeMap<String, (safetensors::Dtype, Vec<usize>, Vec<u8>)>;
fn payloads(c: &PH3ImageCurveStackConfig) -> Payloads {
    let schema = compose::build(c, limits()).unwrap().schema;
    let mut result = Payloads::new();
    for component in schema.components {
        let name = component.external_names[0].clone();
        let layer = name.split('.').nth(1).unwrap().parse::<usize>().unwrap();
        let physical_type = component.physical_element_type();
        let shape = component.dimensions;
        let count = shape.iter().product::<u64>() as usize;
        let (dtype, bytes) = match physical_type {
            ElementType::Bf16 => (
                safetensors::Dtype::BF16,
                (0..count)
                    .flat_map(|i| {
                        bf16::from_f32(if name.contains("q_norm") || name.contains("k_norm") {
                            0.0625 + (i % 3) as f32 / 256.
                        } else {
                            0.75 + (i % 5) as f32 / 32. + layer as f32 / 64.
                        })
                        .to_bits()
                        .to_le_bytes()
                    })
                    .collect(),
            ),
            ElementType::F32 => {
                let values = (0..count).map(|i| {
                    if name.ends_with("weight_s_channel") || name.ends_with("weight_scale") {
                        0.03125 * (1. + (i % 3) as f32 / 8.)
                    } else if name.ends_with("weight_codebook") {
                        i as f32 - 8.
                    } else if name.ends_with("adaln_proj.linear.weight") {
                        ((i * 5 + layer * 3) % 11) as f32 / 1024. - 5. / 1024.
                    } else {
                        let parameter = (i / c.block.hidden_size() as usize) % 6;
                        let modality = i / (6 * c.block.hidden_size()) as usize;
                        match parameter {
                            2 | 5 => {
                                0.15625
                                    + modality as f32 / 32.
                                    + (i % 7) as f32 / 512.
                                    + layer as f32 / 64.
                            }
                            1 | 4 => 0.03125 + (i % 5) as f32 / 1024.,
                            _ => -0.015625 + (i % 9) as f32 / 512.,
                        }
                    }
                });
                (
                    safetensors::Dtype::F32,
                    values.flat_map(f32::to_le_bytes).collect(),
                )
            }
            ElementType::U8 => (safetensors::Dtype::F8_E4M3, vec![0x38; count]),
            ElementType::I8 => {
                let module = name
                    .strip_suffix(".weight")
                    .unwrap()
                    .strip_prefix(&format!("blocks.{layer}."))
                    .unwrap();
                let storage = c
                    .storage_for_layer(layer as u64)
                    .unwrap()
                    .modules()
                    .into_iter()
                    .find(|(name, _)| *name == module)
                    .unwrap()
                    .1;
                if storage == PH3ImageProjectionStorage::Int8W8A8 {
                    let (n, k) = (shape[0] as usize, shape[1] as usize);
                    let phase = name.bytes().map(usize::from).sum::<usize>();
                    let mut values = vec![0u8; count];
                    for row in 0..n {
                        for j in 0..16 {
                            values[row * k + (row * 37 + j * 53 + phase) % k] =
                                (((row + j * 3 + phase) % 7) as i8 - 3) as u8;
                        }
                    }
                    (safetensors::Dtype::I8, values)
                } else {
                    let (n, k) = (shape[0] as usize, shape[1] as usize * 2);
                    let phase = name.bytes().map(usize::from).sum::<usize>();
                    let mut values = vec![0x88; count];
                    for row in 0..n {
                        for j in 0..16 {
                            let col = (row * 37 + j * 53 + phase) % k;
                            let mut code = ((row + j * 3 + phase) % 15) as u8;
                            if code == 8 {
                                code = 15;
                            }
                            let byte = &mut values[row * (k / 2) + col / 2];
                            if col % 2 == 0 {
                                *byte = (*byte & 0xf0) | code;
                            } else {
                                *byte = (*byte & 0x0f) | (code << 4);
                            }
                        }
                    }
                    (safetensors::Dtype::I8, values)
                }
            }
            _ => unreachable!(),
        };
        result.insert(
            name,
            (
                dtype,
                shape.into_iter().map(|n| n as usize).collect(),
                bytes,
            ),
        );
    }
    for layer in 0..c.layers {
        for (name, storage) in c.storage_for_layer(layer).unwrap().modules() {
            let bytes=match storage{PH3ImageProjectionStorage::CodebookW4A8=>br#"{"format":"asym_w4a8_int8","group_size":16,"convrot":true,"convrot_groupsize":256}"#.to_vec(),PH3ImageProjectionStorage::Int8W8A8=>br#"{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}"#.to_vec()};
            result.insert(
                format!("blocks.{layer}.{name}.comfy_quant"),
                (safetensors::Dtype::U8, vec![bytes.len()], bytes),
            );
        }
    }
    let grid = 4097usize;
    let basis = c.block.basis_dim() as usize;
    let table = (0..grid)
        .flat_map(|row| {
            (0..basis).map(move |col| {
                ((row as f32 / (grid - 1) as f32) - 0.5) * (1. + col as f32 / 16.)
                    + col as f32 / 128.
            })
        })
        .flat_map(f32::to_le_bytes)
        .collect();
    result.insert(
        "adaln_t_table".into(),
        (safetensors::Dtype::F32, vec![grid, basis], table),
    );
    result.insert(
        "rope.inv_freq".into(),
        (
            safetensors::Dtype::F32,
            vec![16],
            (0..16)
                .flat_map(|i| (1.0_f32 / (i + 1) as f32).to_le_bytes())
                .collect(),
        ),
    );
    result
}
struct ArchiveFixture {
    _directory: tempfile::TempDir,
    source: ComfyCanonicalWeightSource,
}
fn archive(payloads: &Payloads) -> ArchiveFixture {
    let directory = tempfile::tempdir().unwrap();
    let views = payloads
        .iter()
        .map(|(name, (dtype, shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source =
        ComfyCanonicalWeightSource::new(SafetensorsArchive::open(directory.path()).unwrap())
            .unwrap();
    ArchiveFixture {
        _directory: directory,
        source,
    }
}
#[test]
fn all_layers_load_original_physical_payloads_and_reject_a_late_layer_dtype_error() {
    let s = scene(2, false, false);
    let tensors = payloads(&s.config);
    let files = archive(&tensors);
    let checkpoint = PH3ImageCurveStackCheckpoint::new(&s.config, &files.source, limits()).unwrap();
    assert_eq!(checkpoint.config(), &s.config);
    let schema = compose::build(&s.config, limits()).unwrap().schema;
    for c in &schema.components {
        let payload = checkpoint.source().component(c).unwrap();
        assert_eq!(
            payload.bytes().as_ptr(),
            files
                .source
                .archive()
                .tensor(&c.external_names[0])
                .unwrap()
                .bytes()
                .as_ptr()
        );
    }
    let mut bad = tensors;
    let entry = bad.get_mut("blocks.1.norm1.weight").unwrap();
    entry.0 = safetensors::Dtype::F32;
    entry.2 = vec![0; 4 * s.config.block.hidden_size() as usize];
    let files = archive(&bad);
    assert!(PH3ImageCurveStackCheckpoint::new(&s.config, &files.source, limits()).is_err());
}
#[test]
fn checkpoint_metadata_builds_a_real_w4_then_w8_stack_without_expanding_weights() {
    let mut s = scene(2, false, false);
    s.config = s
        .config
        .with_projection_storage(PH3ImageStackProjectionStorage::PerLayer(vec![
            PH3ImageBlockProjectionStorage::default(),
            PH3ImageBlockProjectionStorage::uniform(PH3ImageProjectionStorage::Int8W8A8),
        ]))
        .unwrap();
    let files = archive(&payloads(&s.config));
    let discovered = PH3ImageCurveStackConfig::from_checkpoint(
        &s.transformer,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        s.config.block.requests().to_vec(),
        &files.source,
        limits(),
    )
    .unwrap();
    assert_eq!(discovered, s.config);
    let checkpoint =
        PH3ImageCurveStackCheckpoint::new(&discovered, &files.source, limits()).unwrap();
    let c = compose::build(&discovered, limits()).unwrap();
    assert_eq!(c.schema.components.len(), 36);
    for spec in &c.schema.components {
        let payload = checkpoint.source().component(spec).unwrap();
        let original = files
            .source
            .archive()
            .tensor(&spec.external_names[0])
            .unwrap();
        assert_eq!(payload.bytes().as_ptr(), original.bytes().as_ptr());
    }
    let layer = PH3ImageCurveBlockConfig::from_checkpoint(
        &s.transformer,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        1,
        3,
        s.config.block.requests().to_vec(),
        &files.source,
    )
    .unwrap();
    assert!(!layer.projection_storage().all_w4());
    super::super::PH3ImageCurveBlockCheckpoint::new(&layer, &files.source, limits().block())
        .unwrap();
}

#[cfg(feature = "cuda")]
mod cuda;
