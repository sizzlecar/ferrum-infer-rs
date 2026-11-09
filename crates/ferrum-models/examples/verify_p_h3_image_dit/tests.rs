use super::*;
use ferrum_models::p_h3_image::PH3ImageTransformerConfig;
use half::bf16;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use serde_json::json;

fn transformer() -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&json!({
        "_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":12,"audio_in_channels":2,"ffn_dim":16,
        "final_norm_eps":1e-5,"freq_dim":4,"hidden_size":8,"in_channels":2,"norm_eps":1e-5,"num_attention_heads":1,
        "num_layers":1,"num_refiner_layers":1,"patch_size":[1,2,2],"qk_norm_eps":1e-5,"rope_freq_dim":2,
        "rope_theta":10000.,"text_dim":4,"time_embed_dim":4,"time_embed_hidden_dim":8
    })).unwrap()).unwrap()
}
fn capture() -> data::Capture {
    serde_json::from_value(json!({"schema":"ferrum.ph3.dit_input.v1","phase":"refined","target_shape":[1,2,1,2,4],"reference_shape":[1,2,1,2,2],"audio_shape":[1,2,2,2],"text_tags":[1,0],"mod_segments":[[0,1,1],[1,2,0],[2,3,3],[3,7,8],[7,9,0]],"layout_segments":[[0,2,"text"],[2,3,"ref_img"],[3,7,"audio"],[7,9,"video"]],"img_update":[false,true,true],"audio_update":[true,true,true,true],"pdd_head_count":1,"block_replacements_present":false,"sigma":1.,"timestep":1000.,"shift_video":1.,"shift_audio":1.,"audio_scale":1.,"denoise_mask_present":false,"audio_denoise_mask_present":false,"checkpoint_sha256":"a".repeat(64)})).unwrap()
}
#[test]
fn checked_routing_preserves_actual_time_rows_and_rejects_different_layout_or_patches() {
    let t = transformer();
    let mut c = capture();
    let (_, times) = c.routing(&t, 3).unwrap();
    assert_eq!(times, [0, 0, 1, 2, 2, 2, 2, 0, 0]);
    assert!(c.routing(&t, 2).is_err());
    c.mod_segments[3][2] = 6;
    assert!(c.routing(&t, 3).is_err());
    c = capture();
    c.img_update[0] = true;
    assert!(c.routing(&t, 3).is_err());
    c = capture();
    c.layout_segments.swap(1, 2);
    assert!(c.routing(&t, 3).is_err());
    c = capture();
    c.block_replacements_present = true;
    assert!(c.routing(&t, 3).is_err());
    c = capture();
    c.pdd_head_count = 2;
    assert!(c.routing(&t, 3).is_err());
    c = capture();
    c.audio_denoise_mask_present = true;
    assert!(c.routing(&t, 3).is_err());
}
#[test]
fn sign_patch_stereo_and_terminal_rounding_use_actual_tensor_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.safetensors");
    let image: Vec<u8> = (0..16)
        .flat_map(|i| bf16::from_f32(-(i as f32)).to_bits().to_le_bytes())
        .collect();
    let audio: Vec<u8> = (0..8)
        .flat_map(|i| bf16::from_f32(-(i as f32)).to_bits().to_le_bytes())
        .collect();
    let expected = [
        0f32, 1., 4., 5., 8., 9., 12., 13., 2., 3., 6., 7., 10., 11., 14., 15.,
    ];
    let positive: Vec<u8> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
    serialize_to_file(
        [
            (
                "image",
                TensorView::new(Dtype::BF16, vec![1, 2, 1, 2, 4], &image).unwrap(),
            ),
            (
                "audio",
                TensorView::new(Dtype::BF16, vec![1, 2, 2, 2], &audio).unwrap(),
            ),
            (
                "raw",
                TensorView::new(Dtype::F32, vec![2, 8], &positive).unwrap(),
            ),
        ],
        &None,
        &path,
    )
    .unwrap();
    let archive = data::open_archive(&path, 4096).unwrap();
    let image = archive.tensor("image").unwrap();
    let raw = archive.tensor("raw").unwrap();
    let mut actual = Vec::new();
    let mut checked = data::TerminalCheck::new(&raw, &mut actual);
    data::reference_image(&image, [1, 2, 1, 2, 4], [1, 2, 2], &mut checked).unwrap();
    checked.finish().unwrap();
    assert_eq!(actual, positive);
    assert!(data::reference_image(&image, [1, 2, 1, 4, 2], [1, 2, 2], std::io::sink()).is_err());
    assert!(data::reference_image(&image, [1, 2, 1, 2, 4], [1, 3, 2], std::io::sink()).is_err());
    let mut wrong = data::TerminalCheck::new(&raw, std::io::sink());
    use std::io::Write;
    assert!(wrong.write_all(&1f32.to_le_bytes()).is_err());
    let mut actual = Vec::new();
    data::reference_audio(&archive.tensor("audio").unwrap(), [1, 2, 2, 2], &mut actual).unwrap();
    assert_eq!(
        actual,
        [0f32, 4., 1., 5., 2., 6., 3., 7.]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
}
#[test]
fn explicit_attention_and_budget_options_do_not_silently_fallback() {
    let args = [
        "--config",
        "config.json",
        "--model-dir",
        "weights",
        "--checkpoint-file",
        "dit.safetensors",
        "--input-bundle",
        "capture.safetensors",
        "--input-metadata",
        "capture.json",
        "--artifact-dir",
        "output",
        "--host-memory-bytes",
        "4096",
        "--attention",
        "sequential",
        "--backend",
        "cuda",
        "--cuda-usable-memory-bytes",
        "4096",
        "--cuda-bootstrap-memory-bytes",
        "1",
        "--cuda-stream-memory-bytes",
        "1",
        "--cuda-opaque-memory-profile",
        "test.v1",
    ];
    let parsed = Arguments::parse(args.map(OsString::from)).unwrap().unwrap();
    assert_eq!(parsed.attention, Attention::Sequential);
    assert_eq!(parsed.input_boundary, InputBoundary::Prepared);
    assert_eq!(parsed.dense_linear_source, DenseLinearSource::Archive);
    assert_eq!(parsed.first_block_taps, FirstBlockTaps::Boundaries);
    assert!(!parsed.capture_first_block);
    let mut enabled = args.map(OsString::from).to_vec();
    enabled.extend(["--capture-first-block".into(), "true".into()]);
    assert!(
        Arguments::parse(enabled)
            .unwrap()
            .unwrap()
            .capture_first_block
    );
    let mut invalid = args.map(OsString::from).to_vec();
    invalid.extend(["--capture-first-block".into(), "1".into()]);
    assert!(Arguments::parse(invalid).is_err());
    for (name, selection) in [
        ("boundaries", FirstBlockTaps::Boundaries),
        ("attention", FirstBlockTaps::Attention),
        ("attention-output", FirstBlockTaps::AttentionOutput),
        ("qkv", FirstBlockTaps::Qkv),
        ("attention-core", FirstBlockTaps::AttentionCore),
    ] {
        let mut explicit = args.map(OsString::from).to_vec();
        explicit.extend(["--first-block-taps".into(), name.into()]);
        assert!(Arguments::parse(explicit.clone()).is_err());
        explicit.extend(["--capture-first-block".into(), "false".into()]);
        assert!(Arguments::parse(explicit).is_err());
        let mut enabled = args.map(OsString::from).to_vec();
        enabled.extend([
            "--capture-first-block".into(),
            "true".into(),
            "--first-block-taps".into(),
            name.into(),
        ]);
        assert_eq!(
            Arguments::parse(enabled).unwrap().unwrap().first_block_taps,
            selection
        );
    }
    let mut invalid = args.map(OsString::from).to_vec();
    invalid.extend([
        "--capture-first-block".into(),
        "true".into(),
        "--first-block-taps".into(),
        "all".into(),
    ]);
    assert!(Arguments::parse(invalid).is_err());
    let mut packed = args.map(OsString::from).to_vec();
    packed.extend(["--input-boundary".into(), "packed".into()]);
    assert_eq!(
        Arguments::parse(packed).unwrap().unwrap().input_boundary,
        InputBoundary::Packed
    );
    let mut mod1 = args.map(OsString::from).to_vec();
    mod1.extend(["--input-boundary".into(), "mod1".into()]);
    assert_eq!(
        Arguments::parse(mod1).unwrap().unwrap().input_boundary,
        InputBoundary::Mod1
    );
    let mut invalid = args.map(OsString::from).to_vec();
    invalid.extend(["--input-boundary".into(), "raw".into()]);
    assert!(Arguments::parse(invalid).is_err());
    assert_eq!(data::input_bindings(InputBoundary::Packed).len(), 4);
    assert_eq!(
        data::taps(InputBoundary::Packed, true, FirstBlockTaps::Boundaries),
        &data::FIRST_BLOCK_TAPS[1..]
    );
    assert!(data::taps(InputBoundary::Packed, false, FirstBlockTaps::Boundaries).is_empty());
    for boundary in [InputBoundary::Prepared, InputBoundary::Packed] {
        assert_eq!(
            data::taps(boundary, true, FirstBlockTaps::Attention),
            &data::FIRST_ATTENTION_TAPS,
        );
        assert!(data::taps(boundary, false, FirstBlockTaps::Attention).is_empty());
    }
    assert_eq!(
        data::input_bindings(InputBoundary::Packed)[0],
        ("packed", data::PACKED_ID)
    );
    assert!(data::FIRST_ATTENTION_TAPS
        .iter()
        .all(|(_, id)| *id != data::PACKED_ID));
    assert_eq!(
        data::taps(InputBoundary::Mod1, true, FirstBlockTaps::Attention),
        &data::FIRST_ATTENTION_TAPS[1..]
    );
    assert_eq!(
        data::taps(InputBoundary::Mod1, true, FirstBlockTaps::Boundaries),
        &data::FIRST_BLOCK_TAPS
    );
    assert_eq!(
        data::input_bindings(InputBoundary::Mod1)[0],
        ("block0_mod1", data::MOD1_ID)
    );
    assert_eq!(
        &data::input_bindings(InputBoundary::Mod1)[1..],
        data::input_bindings(InputBoundary::Prepared)
    );
    for boundary in [
        InputBoundary::Prepared,
        InputBoundary::Packed,
        InputBoundary::Mod1,
    ] {
        let expected = if boundary == InputBoundary::Mod1 {
            &data::FIRST_ATTENTION_OUTPUT_TAPS[1..]
        } else {
            &data::FIRST_ATTENTION_OUTPUT_TAPS[..]
        };
        assert_eq!(
            data::taps(boundary, true, FirstBlockTaps::AttentionOutput),
            expected
        );
        assert!(data::taps(boundary, false, FirstBlockTaps::AttentionOutput).is_empty());
    }
    assert_eq!(
        data::FIRST_ATTENTION_OUTPUT_TAPS[0],
        ("block0_mod1", data::MOD1_ID)
    );
    assert_eq!(
        data::FIRST_ATTENTION_OUTPUT_TAPS[1].0,
        "block0_attention_out"
    );
    assert_eq!(data::FIRST_ATTENTION_OUTPUT_TAPS[2].0, "block0_adaln_rows");
    for boundary in [
        InputBoundary::Prepared,
        InputBoundary::Packed,
        InputBoundary::Mod1,
    ] {
        let expected = if boundary == InputBoundary::Mod1 {
            &data::FIRST_QKV_TAPS[1..]
        } else {
            &data::FIRST_QKV_TAPS[..]
        };
        assert_eq!(data::taps(boundary, true, FirstBlockTaps::Qkv), expected);
        assert!(data::taps(boundary, false, FirstBlockTaps::Qkv).is_empty());
    }
    assert_eq!(data::FIRST_QKV_TAPS[0], ("block0_mod1", data::MOD1_ID));
    assert_eq!(data::FIRST_QKV_TAPS[1].0, "block0_qkv");
    for boundary in [
        InputBoundary::Prepared,
        InputBoundary::Packed,
        InputBoundary::Mod1,
    ] {
        let expected = if boundary == InputBoundary::Mod1 {
            &data::FIRST_ATTENTION_CORE_TAPS[1..]
        } else {
            &data::FIRST_ATTENTION_CORE_TAPS[..]
        };
        assert_eq!(
            data::taps(boundary, true, FirstBlockTaps::AttentionCore),
            expected
        );
        assert!(data::taps(boundary, false, FirstBlockTaps::AttentionCore).is_empty());
    }
    assert_eq!(
        data::FIRST_ATTENTION_CORE_TAPS[0],
        ("block0_mod1", data::MOD1_ID)
    );
    assert_eq!(data::FIRST_ATTENTION_CORE_TAPS[1].0, "block0_attention");
    assert_eq!(parsed.max_input_bytes, 128 << 20);
    assert_eq!(parsed.max_output_bytes, 128 << 20);
    let mut large = args.map(OsString::from).to_vec();
    large.extend([
        "--max-input-bytes".into(),
        "1073741824".into(),
        "--max-output-bytes".into(),
        "1073741824".into(),
    ]);
    let large = Arguments::parse(large).unwrap().unwrap();
    assert_eq!(large.max_input_bytes, 1 << 30);
    assert_eq!(large.max_output_bytes, 1 << 30);
    let mut rounded = args.map(OsString::from).to_vec();
    rounded.extend(["--dense-linear-source".into(), "bf16-roundtrip".into()]);
    assert_eq!(
        Arguments::parse(rounded)
            .unwrap()
            .unwrap()
            .dense_linear_source,
        DenseLinearSource::Bf16Roundtrip
    );
    let mut invalid = args.map(OsString::from).to_vec();
    invalid.extend(["--dense-linear-source".into(), "bf16".into()]);
    assert!(Arguments::parse(invalid).is_err());
    let mut bad = args.map(OsString::from).to_vec();
    bad.extend([
        OsString::from("--cudnn-library"),
        OsString::from("/trusted/libcudnn.so"),
    ]);
    assert!(Arguments::parse(bad).is_err());
    let mut bad = args.map(OsString::from).to_vec();
    bad[15] = "cudnn_sdpa".into();
    assert!(Arguments::parse(bad).is_err());
}

#[test]
fn terminal_boundary_rounds_ties_even_and_preserves_both_zero_signs() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("terminal.safetensors");
    // These exact FP32 midpoints select the even BF16 mantissa on either side.
    // Returned negative-velocity bits are independent explicit oracle values.
    let raw = [
        1.00390625f32,
        1.01171875,
        -1.00390625,
        -1.01171875,
        0.0,
        -0.0,
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect::<Vec<_>>();
    let returned = [0xbf80u16, 0xbf82, 0x3f80, 0x3f82, 0x8000, 0x0000]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "raw",
                TensorView::new(Dtype::F32, vec![6, 1], &raw).unwrap(),
            ),
            (
                "returned",
                TensorView::new(Dtype::BF16, vec![1, 1, 1, 1, 6], &returned).unwrap(),
            ),
        ],
        &None,
        &path,
    )
    .unwrap();
    let archive = data::open_archive(&path, 4096).unwrap();
    let raw = archive.tensor("raw").unwrap();
    let returned = archive.tensor("returned").unwrap();
    let mut output = Vec::new();
    let mut check = data::TerminalCheck::new(&raw, &mut output);
    data::reference_image(&returned, [1, 1, 1, 1, 6], [1, 1, 1], &mut check).unwrap();
    check.finish().unwrap();
    let expected = [1.0f32, 1.015625, -1.0, -1.015625, 0.0, -0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(output, expected);
}

#[test]
fn observed_family_declares_taps_but_preserves_actual_model_graph_weights_and_arithmetic() {
    use ferrum_models::p_h3_image::*;
    use ferrum_runtime::*;
    let t = transformer();
    let config = PH3ImageFinalLayerConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(3, 2).unwrap(),
        1,
        vec![PH3ImageFinalLayerRequest::new(
            1,
            vec![PH3ImageFinalLayerTimeRun::new(4, 0).unwrap()],
            vec![PH3ImageFinalLayerTimeRun::new(1, 0).unwrap()],
        )
        .unwrap()],
    )
    .unwrap();
    let source = TypedFamilyRegistration::new(
        PH3ImageFinalLayerFamily::new(PH3ImageFinalLayerLimits {
            max_metadata_bytes: 1 << 20,
            max_stored_bytes: 1 << 20,
            max_plan_nodes: 1024,
        })
        .unwrap(),
    )
    .prepare_with_profile(
        &serde_json::to_value(&config).unwrap(),
        &NumericalProfileId::new(PH3ImageFinalLayerFamily::CUDA_PROFILE_ID).unwrap(),
    )
    .unwrap();
    let options = config.compile_options().unwrap();
    assert_eq!(
        data::observed_family(
            &source,
            false,
            FirstBlockTaps::Boundaries,
            InputBoundary::Prepared,
            &options,
            DenseLinearSource::Archive
        )
        .unwrap(),
        source
    );
    // An unrelated source cannot silently pretend to contain the selected DiT taps.
    assert!(data::observed_family(
        &source,
        true,
        FirstBlockTaps::Boundaries,
        InputBoundary::Prepared,
        &options,
        DenseLinearSource::Archive
    )
    .is_err());
    assert!(data::observed_family(
        &source,
        false,
        FirstBlockTaps::Boundaries,
        InputBoundary::Packed,
        &options,
        DenseLinearSource::Archive
    )
    .is_err());
    assert_eq!(
        data::installation_family(&source, DenseLinearSource::Archive).unwrap(),
        source
    );
    let rounded = data::installation_family(&source, DenseLinearSource::Bf16Roundtrip).unwrap();
    assert_eq!(rounded.weights(), source.weights());
    assert_eq!(rounded.program().blocks(), source.program().blocks());
    assert_eq!(
        ExecutionWeightPlan::identity(&rounded).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_ne!(
        rounded.fingerprint().unwrap(),
        source.fingerprint().unwrap()
    );
    assert_eq!(
        rounded.canonical_config()["dense_linear_source"],
        "bf16_roundtrip"
    );
    let tap = source
        .numerical_profile()
        .boundaries
        .iter()
        .find(|(id, dtype)| {
            **dtype == ElementType::Bf16 && !source.program().outputs().contains(id)
        })
        .map(|(id, _)| id.clone())
        .unwrap();
    let observed =
        data::prepare_observation(&source, vec![tap.clone()], FirstBlockTaps::Boundaries).unwrap();
    let attention =
        data::prepare_observation(&source, vec![tap.clone()], FirstBlockTaps::Attention).unwrap();
    let attention_output =
        data::prepare_observation(&source, vec![tap.clone()], FirstBlockTaps::AttentionOutput)
            .unwrap();
    let qkv = data::prepare_observation(&source, vec![tap.clone()], FirstBlockTaps::Qkv).unwrap();
    let attention_core =
        data::prepare_observation(&source, vec![tap.clone()], FirstBlockTaps::AttentionCore)
            .unwrap();
    assert_eq!(attention_core.program().blocks(), source.program().blocks());
    assert_eq!(attention_core.program().inputs(), source.program().inputs());
    assert_eq!(attention_core.program().outputs(), qkv.program().outputs());
    assert_eq!(
        ExecutionWeightPlan::identity(&attention_core).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_eq!(
        attention_core.canonical_config()["first_block_taps"],
        "attention_core"
    );
    assert_ne!(
        attention_core.fingerprint().unwrap(),
        qkv.fingerprint().unwrap()
    );
    assert_ne!(
        attention_core.numerical_profile().id,
        qkv.numerical_profile().id
    );
    assert_eq!(qkv.program().blocks(), source.program().blocks());
    assert_eq!(qkv.program().inputs(), source.program().inputs());
    assert_eq!(qkv.program().outputs(), attention.program().outputs());
    assert_eq!(
        ExecutionWeightPlan::identity(&qkv).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_eq!(qkv.canonical_config()["first_block_taps"], "qkv");
    assert_ne!(qkv.fingerprint().unwrap(), attention.fingerprint().unwrap());
    assert_ne!(qkv.numerical_profile().id, attention.numerical_profile().id);
    assert_eq!(
        attention_output.program().blocks(),
        source.program().blocks()
    );
    assert_eq!(
        attention_output.program().inputs(),
        source.program().inputs()
    );
    assert_eq!(
        attention_output.program().outputs(),
        attention.program().outputs()
    );
    assert_eq!(
        ExecutionWeightPlan::identity(&attention_output).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_eq!(
        attention_output.canonical_config()["first_block_taps"],
        "attention_output"
    );
    assert_ne!(
        attention_output.fingerprint().unwrap(),
        attention.fingerprint().unwrap()
    );
    assert_ne!(
        attention_output.numerical_profile().id,
        attention.numerical_profile().id
    );
    assert_eq!(attention.program().blocks(), source.program().blocks());
    assert_eq!(attention.program().weights(), source.program().weights());
    assert_eq!(attention.weights(), source.weights());
    assert_eq!(attention.program().outputs(), observed.program().outputs());
    assert_eq!(
        attention.canonical_config()["first_block_taps"],
        "attention"
    );
    assert!(observed
        .canonical_config()
        .get("first_block_taps")
        .is_none());
    assert_ne!(
        attention.fingerprint().unwrap(),
        observed.fingerprint().unwrap()
    );
    assert_ne!(
        attention.numerical_profile().id,
        observed.numerical_profile().id
    );
    assert_eq!(
        ExecutionWeightPlan::identity(&attention).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_eq!(observed.weights(), source.weights());
    assert_eq!(observed.program().inputs(), source.program().inputs());
    assert_eq!(
        observed.program().input_tensor_specs(),
        source.program().input_tensor_specs()
    );
    assert_eq!(observed.program().blocks(), source.program().blocks());
    assert_eq!(observed.program().states(), source.program().states());
    assert_eq!(observed.program().weights(), source.program().weights());
    let mut expected = source.program().outputs().to_vec();
    expected.push(tap.clone());
    assert_eq!(observed.program().outputs(), expected);
    assert_eq!(
        observed.numerical_profile().operations,
        source.numerical_profile().operations
    );
    assert_eq!(
        observed.numerical_profile().boundaries,
        source.numerical_profile().boundaries
    );
    assert_eq!(
        ExecutionWeightPlan::identity(&observed).unwrap(),
        ExecutionWeightPlan::identity(&source).unwrap()
    );
    assert_ne!(observed.config_fingerprint(), source.config_fingerprint());
    assert_ne!(
        observed.fingerprint().unwrap(),
        source.fingerprint().unwrap()
    );
    assert_eq!(
        observed.canonical_config()["taps"],
        serde_json::to_value([tap.clone()]).unwrap()
    );
    assert_eq!(
        observed.canonical_config()["source_family_fingerprint"],
        source.fingerprint().unwrap()
    );
    assert!(
        data::prepare_observation(&source, vec![tap.clone(), tap], FirstBlockTaps::Boundaries)
            .is_err()
    );
    assert!(data::prepare_observation(
        &source,
        vec![source.program().outputs()[0].clone()],
        FirstBlockTaps::Boundaries
    )
    .is_err());
    assert_eq!(
        data::execution_options(&options, &observed, InputBoundary::Prepared).unwrap(),
        options
    );
    let mut hinted = options.clone();
    hinted.retain_completion_value(observed.program().outputs()[0].clone());
    assert!(data::execution_options(&hinted, &observed, InputBoundary::Packed).is_err());
}

#[test]
fn packed_cut_uses_dependency_closure_and_preserves_original_nodes_and_full_weights() {
    use ferrum_runtime::*;
    let value = |name| ProgramValueId::new(format!("value.{name}")).unwrap();
    let family = ModelFamilyId::new("family.actual_dit.cut_fixture").unwrap();
    let tensor = ProgramTensorSpec {
        dimensions: vec![2, 4],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let raw = value("raw");
    let basis = value("basis");
    let cos = value("cos");
    let sin = value("sin");
    let weight = value("weight");
    let packed = value("packed");
    let hidden = value("block0");
    let image = value("image");
    let audio = value("audio");
    let node = |name: &str, inputs, output| ProgramNode {
        id: NodeId::new(format!("node.{name}")).unwrap(),
        operation_id: OperationId::new(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID).unwrap(),
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes: BTreeMap::new(),
    };
    // The prefix consumes a weight that must remain installed after its node is removed.
    let prefix = node("prefix", vec![raw.clone(), weight.clone()], packed.clone());
    let main = node("main", vec![packed.clone(), basis.clone()], hidden.clone());
    let image_head = node("image", vec![hidden.clone(), cos.clone()], image.clone());
    let audio_head = node("audio", vec![hidden.clone(), sin.clone()], audio.clone());
    let inputs = vec![raw.clone(), basis.clone(), cos.clone(), sin.clone()];
    let source = ModelProgram::new(
        family.clone(),
        inputs.clone(),
        vec![ProgramBlock {
            id: "block.original".into(),
            nodes: vec![prefix, main.clone(), image_head.clone(), audio_head.clone()],
        }],
        vec![],
        vec![WeightReference {
            weight_id: WeightId::new("weight.prefix").unwrap(),
            value_id: weight,
            tensor: tensor.clone(),
        }],
        vec![image.clone(), audio.clone()],
    )
    .unwrap()
    .with_input_tensor_specs(inputs.into_iter().map(|id| (id, tensor.clone())).collect())
    .unwrap();
    let cut = data::cut_at_input(
        &source,
        family.clone(),
        &packed,
        tensor.clone(),
        &[hidden.clone()],
    )
    .unwrap();
    assert_eq!(cut.inputs(), [packed.clone(), basis, cos, sin]);
    assert_eq!(cut.blocks()[0].nodes, [main, image_head, audio_head]);
    assert_eq!(cut.outputs(), [image, audio, hidden.clone()]);
    assert_eq!(cut.weights(), source.weights());
    assert!(!cut.input_tensor_specs().contains_key(&raw));
    assert_eq!(cut.input_tensor_specs().get(&packed), Some(&tensor));
    assert!(data::cut_at_input(&source, family.clone(), &raw, tensor.clone(), &[]).is_err());
    assert!(data::cut_at_input(
        &source,
        family.clone(),
        &packed,
        tensor.clone(),
        &[packed.clone()]
    )
    .is_err());
    assert!(
        data::cut_at_input(&source, family, &packed, tensor, &[hidden.clone(), hidden]).is_err()
    );
}

#[test]
fn mod1_cut_preserves_packed_residual_branch_and_the_six_original_inputs() {
    use ferrum_runtime::*;
    let value = |name| ProgramValueId::new(name).unwrap();
    let family = ModelFamilyId::new("family.actual_dit.mod1_fixture").unwrap();
    let tensor = ProgramTensorSpec {
        dimensions: vec![2, 4],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let inputs = data::input_bindings(InputBoundary::Prepared)
        .iter()
        .map(|(_, id)| value(*id))
        .collect::<Vec<_>>();
    let packed = value(data::PACKED_ID);
    let mod1 = value(data::MOD1_ID);
    let residual = value(data::FIRST_ATTENTION_TAPS[1].1);
    let weight = value("value.fixture.weight");
    let gate = value("value.fixture.gate");
    let norm = value("value.fixture.norm");
    let attention = value("value.fixture.attention");
    let image = value("value.fixture.image");
    let audio = value("value.fixture.audio");
    let node = |name: &str, inputs, output| ProgramNode {
        id: NodeId::new(format!("node.{name}")).unwrap(),
        operation_id: OperationId::new(PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID).unwrap(),
        required_version: ContractVersion::new(1, 0),
        work: ProgramNodeWorkSpec::Fixed,
        inputs,
        outputs: vec![output],
        attributes: BTreeMap::new(),
    };
    // This fixture tests graph reachability, not the synthetic nodes' arithmetic.
    let prefix = node(
        "prefix",
        vec![
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
            weight.clone(),
        ],
        packed.clone(),
    );
    let gate_node = node(
        "gate",
        vec![inputs[3].clone(), weight.clone()],
        gate.clone(),
    );
    let norm_node = node("norm", vec![packed.clone()], norm.clone());
    let mod_node = node("mod", vec![norm, gate.clone()], mod1.clone());
    let attention_node = node(
        "attention",
        vec![mod1.clone(), inputs[4].clone(), inputs[5].clone()],
        attention.clone(),
    );
    let residual_node = node("residual", vec![packed, gate, attention], residual.clone());
    let image_head = node("image", vec![residual.clone()], image.clone());
    let audio_head = node("audio", vec![residual.clone()], audio.clone());
    let retained = vec![
        prefix.clone(),
        gate_node.clone(),
        attention_node.clone(),
        residual_node.clone(),
        image_head.clone(),
        audio_head.clone(),
    ];
    let source = ModelProgram::new(
        family.clone(),
        inputs.clone(),
        vec![ProgramBlock {
            id: "block.original".into(),
            nodes: vec![
                prefix,
                gate_node,
                norm_node,
                mod_node,
                attention_node,
                residual_node,
                image_head,
                audio_head,
            ],
        }],
        vec![],
        vec![WeightReference {
            weight_id: WeightId::new("weight.fixture").unwrap(),
            value_id: weight,
            tensor: tensor.clone(),
        }],
        vec![image.clone(), audio.clone()],
    )
    .unwrap()
    .with_input_tensor_specs(
        inputs
            .iter()
            .cloned()
            .map(|id| (id, tensor.clone()))
            .collect(),
    )
    .unwrap();
    let cut =
        data::cut_at_input(&source, family, &mod1, tensor.clone(), &[residual.clone()]).unwrap();
    let expected_inputs = data::input_bindings(InputBoundary::Mod1)
        .iter()
        .map(|(_, id)| value(*id))
        .collect::<Vec<_>>();
    assert_eq!(cut.inputs(), expected_inputs);
    assert_eq!(cut.blocks()[0].nodes, retained);
    assert_eq!(cut.outputs(), [image, audio, residual]);
    assert!(!cut.outputs().contains(&mod1));
    assert_eq!(cut.weights(), source.weights());
    assert_eq!(cut.states(), source.states());
    for id in inputs.iter().chain(std::iter::once(&mod1)) {
        assert_eq!(cut.input_tensor_specs().get(id), Some(&tensor));
    }
}

#[test]
fn tap_extents_follow_original_specs_including_compact_adaln_rows() {
    use ferrum_runtime::*;
    let make = |dimensions, element_type| ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let hidden = ProgramValueId::new(data::FIRST_ATTENTION_OUTPUT_TAPS[1].1).unwrap();
    let adaln = ProgramValueId::new(data::FIRST_ATTENTION_OUTPUT_TAPS[2].1).unwrap();
    let qkv = ProgramValueId::new(data::FIRST_QKV_TAPS[1].1).unwrap();
    let attention = ProgramValueId::new(data::FIRST_ATTENTION_CORE_TAPS[1].1).unwrap();
    let options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (hidden.clone(), make(vec![17746, 5376], ElementType::Bf16)),
        (adaln.clone(), make(vec![6, 32256], ElementType::Bf16)),
        (qkv.clone(), make(vec![17746, 21504], ElementType::Bf16)),
        (
            attention.clone(),
            make(vec![17746, 7168], ElementType::Bf16),
        ),
    ]))
    .unwrap();
    let hidden_spec = data::tap_tensor_spec(&options, hidden.as_str()).unwrap();
    let adaln_spec = data::tap_tensor_spec(&options, adaln.as_str()).unwrap();
    assert_eq!(hidden_spec.dimensions(), [17746, 5376]);
    assert_eq!(hidden_spec.minimum_storage_bytes().unwrap(), 190804992);
    assert_eq!(adaln_spec.dimensions(), [6, 32256]);
    assert_eq!(adaln_spec.minimum_storage_bytes().unwrap(), 387072);
    let qkv_spec = data::tap_tensor_spec(&options, qkv.as_str()).unwrap();
    assert_eq!(qkv_spec.dimensions(), [17746, 21504]);
    assert_eq!(qkv_spec.minimum_storage_bytes().unwrap(), 763219968);
    let attention_spec = data::tap_tensor_spec(&options, attention.as_str()).unwrap();
    assert_eq!(attention_spec.dimensions(), [17746, 7168]);
    assert_eq!(attention_spec.minimum_storage_bytes().unwrap(), 254406656);
    assert!(data::tap_tensor_spec(&options, data::MOD1_ID).is_err());
    for invalid in [
        make(vec![6, 32256], ElementType::F32),
        make(vec![6, 6, 5376], ElementType::Bf16),
    ] {
        let options =
            ProgramPlanCompileOptions::new(BTreeMap::from([(adaln.clone(), invalid)])).unwrap();
        assert!(data::tap_tensor_spec(&options, adaln.as_str()).is_err());
    }
}
