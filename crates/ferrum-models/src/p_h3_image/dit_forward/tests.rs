use super::*;
use half::bf16;
#[cfg(feature = "cuda")]
mod cuda;
mod edit_prefix_oracle;
mod header;
pub(in crate::p_h3_image) use super::test_fixture as shared_fixture;
pub(in crate::p_h3_image) use super::test_fixture::{limits, transformer};
pub(super) fn config(batch: usize) -> PH3ImageDiTConfig {
    let t = transformer();
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = PH3ImageReferenceImage::new(2, 4).unwrap();
    let tags = [
        PH3ImageModality::Image,
        PH3ImageModality::Text,
        PH3ImageModality::Text,
    ];
    let requests = (0..batch)
        .map(|i| {
            let refs = if i % 2 == 0 { Some(reference) } else { None };
            let layout = PH3ImageRequestLayout::new(&t, &tags, target, refs.as_slice()).unwrap();
            let mut times = vec![(i % 2) as u64; layout.rows()];
            times[layout.reference_rows()].fill(2);
            PH3ImageDiTRequest::new(&t, &tags, target, refs, &times, 1 << 20).unwrap()
        })
        .collect();
    PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        requests,
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::PerLayer(vec![
            PH3ImageBlockProjectionStorage::default(),
            PH3ImageBlockProjectionStorage::uniform(
                super::super::PH3ImageProjectionStorage::Int8W8A8,
            ),
        ]),
    )
    .unwrap()
}
#[test]
fn explicit_sdpa_profiles_preserve_payloads_and_disclose_vendor_arithmetic() {
    for phase in [
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageDiTPhase::RefineOnly,
        PH3ImageDiTPhase::RefinedText,
    ] {
        let sequential = config(2).with_phase(phase);
        let vendor = sequential
            .clone()
            .with_attention_policy(PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1)
            .unwrap();
        let a = program::compose(&sequential, limits()).unwrap();
        let b = program::compose(&vendor, limits()).unwrap();
        assert_eq!(a.schema, b.schema);
        assert_eq!(
            serde_json::to_value(&a.program).unwrap(),
            serde_json::to_value(&b.program).unwrap()
        );
        assert_eq!(a.profile.boundaries, b.profile.boundaries);
        assert_ne!(a.profile.id, b.profile.id);
        let mut seen = false;
        for (old, new) in a.profile.operations.iter().zip(&b.profile.operations) {
            if old.operation_id.as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID {
                seen = true;
                assert_eq!(old.multiplication_type, Some(ElementType::F32));
                assert_eq!(old.accumulation_type, Some(ElementType::F32));
                assert_eq!(new.multiplication_type, None);
                assert_eq!(new.accumulation_type, None);
            } else {
                assert_eq!(old, new);
            }
        }
        assert!(seen);
        b.profile.validate_program(&b.program).unwrap();
    }
}

#[test]
fn explicit_sdpa_refinement_keys_cannot_cross_attention_arithmetic() {
    let sequential = config(2);
    let vendor = sequential
        .clone()
        .with_attention_policy(PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1)
        .unwrap();
    let host = account(1 << 20);
    let connector = f32_tensor(
        &host,
        "value.connector",
        vec![6, 4],
        (0..24).map(|i| i as f32 / 32.),
    );
    let sequential_keys = cache::fixture_keys(&sequential, &connector);
    let vendor_keys = cache::fixture_keys(&vendor, &connector);
    cache::validate_keys(&sequential, &sequential_keys).unwrap();
    cache::validate_keys(&vendor, &vendor_keys).unwrap();
    assert!(cache::validate_keys(&vendor, &sequential_keys).is_err());
    assert!(cache::validate_keys(&sequential, &vendor_keys).is_err());
}

#[test]
fn first_and_steady_share_complete_original_weights_and_all_node_geometry() {
    let first = config(2);
    let steady = first.clone().with_phase(PH3ImageDiTPhase::RefinedText);
    let a = program::compose(&first, limits()).unwrap();
    let b = program::compose(&steady, limits()).unwrap();
    assert_eq!(a.schema, b.schema);
    assert_eq!(a.program.weights(), b.program.weights());
    assert_eq!(a.schema.components.len(), 22 + 14 + 27 + 7 + 4);
    let names = a
        .schema
        .components
        .iter()
        .flat_map(|v| &v.external_names)
        .collect::<BTreeSet<_>>();
    assert!(names.contains(&"video_patch_proj.weight".to_string()));
    assert!(names.contains(&"audio_patch_proj.bias".to_string()));
    assert!(names.contains(&"token_refiner.blocks.1.attn.qkv_proj.weight".to_string()));
    assert!(a.program.blocks()[0]
        .nodes
        .iter()
        .any(|n| n.id.as_str().starts_with("node.p_h3_image.token_refiner.")));
    assert!(!b.program.blocks()[0]
        .nodes
        .iter()
        .any(|n| n.id.as_str().starts_with("node.p_h3_image.token_refiner.")));
    assert_eq!(a.program.outputs().len(), 3);
    assert_eq!(b.program.outputs().len(), 2);
    for body in [&a, &b] {
        body.schema
            .validate_program_references(body.program.family_id(), &body.program)
            .unwrap();
        body.profile.validate_program(&body.program).unwrap();
        let expected = body
            .program
            .inputs()
            .iter()
            .chain(
                body.program
                    .blocks()
                    .iter()
                    .flat_map(|b| &b.nodes)
                    .flat_map(|n| &n.outputs),
            )
            .collect::<BTreeSet<_>>();
        assert_eq!(expected, body.options.tensor_specs().keys().collect());
        assert!(body.options.completion_retention().is_empty());
    }
    let first_norm = a.program.blocks()[0]
        .nodes
        .iter()
        .find(|n| {
            n.id.as_str().contains("curve_stack.layer0")
                && n.operation_id.as_str() == ROW_RMS_NORM_SINGLE_ROUND_BF16_OPERATION_ID
        })
        .unwrap();
    assert_eq!(first_norm.inputs[0].as_str(), program::PACKED);
    let final_node = a.program.blocks()[0]
        .nodes
        .iter()
        .find(|n| n.id.as_str().ends_with("final_layer.image.chunk0.gather"))
        .unwrap();
    assert_eq!(
        final_node.inputs[0].as_str(),
        PH3ImageCurveStackFamily::OUTPUT_ID
    );
}
#[test]
fn request_pack_order_crosses_transport_run_capacity_without_cross_document_attention() {
    // Even a request without a reference has three runs. Derive the
    // document count from the actual transport contract's run capacity.
    let c = config(ROW_GATHER_BF16_MAX_RUNS / 3 + 1);
    let runs = program::packing_runs(&c).unwrap();
    assert!(runs.len() > ROW_GATHER_BF16_MAX_RUNS);
    let source_rows = c.text_rows().unwrap() + c.image_rows().unwrap() + c.audio_rows().unwrap();
    let source = (0..source_rows).collect::<Vec<_>>();
    let actual = runs
        .iter()
        .flat_map(|&(start, len)| {
            source[start as usize..(start + len) as usize]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let mut text = 0u64;
    let mut image = c.text_rows().unwrap();
    let mut audio = image + c.image_rows().unwrap();
    let mut expected = vec![];
    for request in &c.requests {
        expected.extend(text..text + request.text_rows);
        let refs = request.reference_rows(&c).unwrap();
        let targets = request.target_rows(&c).unwrap();
        expected.extend(image..image + refs);
        expected.extend(audio..audio + 4);
        expected.extend(image + refs..image + refs + targets);
        text += request.text_rows;
        image += refs + targets;
        audio += 4;
    }
    assert_eq!(actual, expected);
    assert_eq!(
        actual.iter().copied().collect::<BTreeSet<_>>().len(),
        source_rows as usize
    );
    let body = program::compose(&c, limits()).unwrap();
    let pack = body.program.blocks()[0]
        .nodes
        .iter()
        .filter(|n| {
            n.id.as_str()
                .starts_with("node.p_h3_image.dit_forward.pack")
        })
        .collect::<Vec<_>>();
    assert_eq!(pack.len(), 3);
    for node in body.program.blocks()[0]
        .nodes
        .iter()
        .filter(|n| n.operation_id.as_str() == ROW_GATHER_BF16_OPERATION_ID)
    {
        assert!(body.options.tensor_specs().contains_key(&node.outputs[0]));
    }
    // The exact stack/refiner programs carry their request segments; packing
    // adds transport only, never an attention that joins request documents.
    assert_eq!(
        body.program.blocks()[0]
            .nodes
            .iter()
            .filter(|n| n.operation_id.as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID)
            .count(),
        4
    );
}
#[test]
fn complete_composition_rejects_real_node_weight_and_metadata_capacity() {
    let c = config(2);
    let body = program::compose(&c, limits()).unwrap();
    let bytes = body
        .schema
        .components
        .iter()
        .map(|c| c.physical_bytes().unwrap())
        .sum::<u64>();
    let nodes = body.program.blocks()[0].nodes.len();
    for cap in [
        PH3ImageDiTLimits {
            max_stored_bytes: bytes - 1,
            ..limits()
        },
        PH3ImageDiTLimits {
            max_plan_nodes: nodes - 1,
            ..limits()
        },
        PH3ImageDiTLimits {
            max_metadata_bytes: 64,
            ..limits()
        },
    ] {
        assert!(program::compose(&c, cap).is_err());
    }
    let mut malformed = c.clone();
    malformed.requests[0].target.height = 3;
    assert!(program::compose(&malformed, limits()).is_err());
    malformed = c;
    malformed.requests[0].text_rows += 1;
    assert!(program::compose(&malformed, limits()).is_err());
    let mut wrong_segments = serde_json::to_value(config(2)).unwrap();
    wrong_segments["refiner"]["requests"] = serde_json::json!([2, 4]);
    let wrong_segments: PH3ImageDiTConfig = serde_json::from_value(wrong_segments).unwrap();
    assert!(program::compose(&wrong_segments, limits()).is_err());
}
#[test]
fn request_crosswalk_rejects_independent_modality_and_final_time_changes() {
    let c = config(2);
    let mut start = 0u64;
    let starts = c.requests[0]
        .block
        .runs()
        .iter()
        .map(|r| {
            let s = start;
            start += r.length();
            s
        })
        .collect::<Vec<_>>();
    let reference = c.requests[0].text_rows;
    let audio = reference + c.requests[0].reference_rows(&c).unwrap();
    let image = audio + 4;
    for (row, modality) in [(0, 1), (reference, 1), (audio, 0), (image, 1)] {
        let mut raw = serde_json::to_value(&c).unwrap();
        let index = starts.iter().rposition(|&s| s <= row).unwrap();
        raw["requests"][0]["block"]["runs"][index]["modality"] = serde_json::json!(modality);
        raw["stack"]["block"]["requests"][0] = raw["requests"][0]["block"].clone();
        let bad: PH3ImageDiTConfig = serde_json::from_value(raw).unwrap();
        assert!(
            program::compose(&bad, limits()).is_err(),
            "modality at row {row}"
        );
    }
    for stream in ["audio", "image"] {
        let mut raw = serde_json::to_value(&c).unwrap();
        raw["requests"][0]["final_layer"][stream][0]["time_index"] = serde_json::json!(1);
        raw["final_layer"]["requests"][0] = raw["requests"][0]["final_layer"].clone();
        let bad: PH3ImageDiTConfig = serde_json::from_value(raw).unwrap();
        assert!(
            program::compose(&bad, limits()).is_err(),
            "independent final {stream} time"
        );
    }
    // A real run may span presentation-image and reference-image boundaries.
    // The crosswalk advances RLE cursors rather than demanding equal run cuts.
    let t = transformer();
    let tags = [PH3ImageModality::Image; 3];
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let refs = Some(PH3ImageReferenceImage::new(2, 4).unwrap());
    let layout = PH3ImageRequestLayout::new(&t, &tags, target, refs.as_slice()).unwrap();
    let request =
        PH3ImageDiTRequest::new(&t, &tags, target, refs, &vec![0; layout.rows()], 1 << 20).unwrap();
    let merged = PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        vec![request],
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::default(),
    )
    .unwrap();
    program::compose(&merged, limits()).unwrap();
}
fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn f32_tensor(
    host: &HostMemoryAccount,
    id: &str,
    shape: Vec<u64>,
    values: impl Iterator<Item = f32>,
) -> ManagedHostTensor {
    let data = values.flat_map(f32::to_le_bytes).collect::<Vec<_>>();
    let mut a = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, data.len())
        .unwrap();
    a.as_mut().copy_from_slice(&data);
    inputs::managed(id, config::tensor(shape, ElementType::F32), a.freeze()).unwrap()
}
fn supplemental(
    c: &PH3ImageDiTConfig,
    h: &HostMemoryAccount,
) -> (ManagedHostTensor, ManagedHostTensor, ManagedHostTensor) {
    let specs = c.input_tensor_specs().unwrap();
    let [a, b, c] = [
        PH3ImageDiTFamily::BASIS_ID,
        PH3ImageDiTFamily::COSINE_ID,
        PH3ImageDiTFamily::SINE_ID,
    ]
    .map(|id| {
        let spec = &specs[&ProgramValueId::new(id).unwrap()];
        let mut allocation = h
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                spec.byte_len().unwrap() as usize,
            )
            .unwrap();
        if spec.element_type == ElementType::Bf16 {
            for b in allocation.as_mut().chunks_exact_mut(2) {
                b.copy_from_slice(&bf16::from_f32(0.25).to_bits().to_le_bytes());
            }
        }
        inputs::managed(id, spec.clone(), allocation.freeze()).unwrap()
    });
    (a, b, c)
}
#[test]
fn managed_patch_adapter_preserves_f32_solver_and_separate_reference_boundary() {
    let c = config(2);
    let host = account(1 << 20);
    let images = (0..2)
        .map(|request| {
            f32_tensor(
                &host,
                "value.sample.image",
                vec![1, 2, 1, 4, 4],
                (0..32).map(|i| 1.00390625 + i as f32 / 32. + request as f32),
            )
        })
        .collect::<Vec<_>>();
    let audio = (0..2)
        .map(|request| {
            f32_tensor(
                &host,
                "value.sample.audio",
                vec![1, 2, 2, 2],
                (0..8).map(|i| 1.00390625 + i as f32 / 32. + request as f32),
            )
        })
        .collect::<Vec<_>>();
    let reference = f32_tensor(
        &host,
        "value.reference",
        vec![1, 2, 1, 2, 4],
        (0..16).map(|i| 1.00390625 + i as f32 / 32.),
    );
    let noise = f32_tensor(
        &host,
        "value.noise.patch_order",
        vec![2, 8],
        std::iter::repeat_n(-0.25, 16),
    );
    let connector = f32_tensor(
        &host,
        "value.connector",
        vec![6, 4],
        (0..24).map(|i| 1.00390625 + i as f32 / 32.),
    );
    let keys = cache::fixture_keys(&c, &connector);
    let original = images[0].bytes().bytes().to_vec();
    let samples = (0..2)
        .map(|i| PH3ImageDiTSample {
            image: &images[i],
            audio: &audio[i],
        })
        .collect::<Vec<_>>();
    let references = [
        Some(PH3ImageDiTReference::default_augmented(&reference, &noise)),
        None,
    ];
    let (basis, cos, sin) = supplemental(&c, &host);
    let input = PH3ImageDiTInputAdapter::new(&c, &host)
        .unwrap()
        .prepare(
            &samples,
            &references,
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys,
            },
            basis,
            cos,
            sin,
        )
        .unwrap();
    assert_eq!(images[0].bytes().bytes(), original);
    assert_eq!(input.inputs.len(), 6);
    assert_eq!(input.conditioning.as_ref().unwrap(), &keys);
    let image = input
        .inputs
        .iter()
        .find(|x| x.value_id().as_str() == PH3ImageDiTFamily::IMAGE_PATCHES_ID)
        .unwrap();
    let image_values = image
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    // Independent explicit first patch, channels before the two spatial axes.
    for (column, source) in [0, 1, 4, 5, 8, 9, 12, 13].into_iter().enumerate() {
        let raw = 1.00390625 + source as f32 / 32.;
        let a = f32::from_bits(0x3f7f_be77) * raw;
        let b = f32::from_bits(0x3a83_126f) * (-0.25);
        assert_eq!(image_values[column].to_bits(), (a + b).to_bits());
    }
    assert_ne!(image_values[0], bf16::from_f32(image_values[0]).to_f32());
    for (column, source) in [0, 1, 4, 5, 16, 17, 20, 21].into_iter().enumerate() {
        assert_eq!(
            image_values[16 + column],
            bf16::from_f32(1.00390625 + source as f32 / 32.).to_f32()
        );
    }
    let audio_view = input
        .inputs
        .iter()
        .find(|x| x.value_id().as_str() == PH3ImageDiTFamily::AUDIO_PATCHES_ID)
        .unwrap();
    let v = audio_view
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(v[0], 1.);
    assert_eq!(v[1], bf16::from_f32(1.00390625 + 4. / 32.).to_f32());
    assert_eq!(v[8], bf16::from_f32(2.00390625).to_f32());
    let text = input
        .inputs
        .iter()
        .find(|x| x.value_id().as_str() == PH3ImageDiTFamily::RAW_TEXT_ID)
        .unwrap();
    assert_eq!(
        &text.bytes().bytes()[..2],
        &bf16::from_f32(1.00390625).to_bits().to_le_bytes()
    );
    drop(input);
    drop(samples);
    drop(keys);
    drop(images);
    drop(audio);
    drop(reference);
    drop(noise);
    drop(connector);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

fn augmented_reference_values(
    augmentation: f64,
    value: f32,
    noise: Option<f32>,
) -> anyhow::Result<Vec<f32>> {
    let c = config(1);
    let host = account(1 << 20);
    let result = (|| {
        let image = f32_tensor(
            &host,
            "value.sample",
            vec![1, 2, 1, 4, 4],
            std::iter::repeat_n(0.0, 32),
        );
        let audio = f32_tensor(
            &host,
            "value.audio",
            vec![1, 2, 2, 2],
            std::iter::repeat_n(0.0, 8),
        );
        let reference = f32_tensor(
            &host,
            "value.reference",
            vec![1, 2, 1, 2, 4],
            std::iter::repeat_n(value, 16),
        );
        let noise = noise.map(|v| {
            f32_tensor(
                &host,
                "value.reference.noise",
                vec![2, 8],
                std::iter::repeat_n(v, 16),
            )
        });
        let connector = f32_tensor(
            &host,
            "value.connector",
            vec![3, 4],
            std::iter::repeat_n(0.0, 12),
        );
        let keys = cache::fixture_keys(&c, &connector);
        let (basis, cos, sin) = supplemental(&c, &host);
        let prepared = PH3ImageDiTInputAdapter::new(&c, &host)?.prepare(
            &[PH3ImageDiTSample {
                image: &image,
                audio: &audio,
            }],
            &[Some(PH3ImageDiTReference {
                latent: &reference,
                patch_noise: noise.as_ref(),
                augmentation,
            })],
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys,
            },
            basis,
            cos,
            sin,
        )?;
        let patches = prepared
            .inputs
            .iter()
            .find(|t| t.value_id().as_str() == PH3ImageDiTFamily::IMAGE_PATCHES_ID)
            .unwrap();
        Ok(patches.bytes().bytes()[..reference.bytes().len()]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    })();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    result
}

#[test]
fn reference_augmentation_preserves_python_scalar_coefficients_at_f32_boundary() {
    // Literal IEEE-754 expectations: isolate each coefficient through the
    // real managed adapter instead of recomputing its subtraction as an oracle.
    for (augmentation, signal_bits, noise_bits) in [
        (0.0, 0x0000_0000, 0x3f80_0000),
        (1.0, 0x3f80_0000, 0x0000_0000),
        (0.999, 0x3f7f_be77, 0x3a83_126f),
        (0.5, 0x3f00_0000, 0x3f00_0000),
        (2.0_f64.powi(-25), 0x3300_0000, 0x3f80_0000),
        (2.0_f64.powi(-149), 0x0000_0001, 0x3f80_0000),
        (2.0_f64.powi(-150), 0x0000_0000, 0x3f80_0000),
        (f64::from_bits(1), 0x0000_0000, 0x3f80_0000),
        // The nearest f64 below one still needs noise even though its signal
        // coefficient rounds to exactly one in F32; its complement is 2^-53.
        (
            f64::from_bits(1.0_f64.to_bits() - 1),
            0x3f80_0000,
            0x2500_0000,
        ),
    ] {
        for (value, noise, expected) in [(1.0, 0.0, signal_bits), (0.0, 1.0, noise_bits)] {
            let actual = augmented_reference_values(augmentation, value, Some(noise)).unwrap();
            assert!(
                actual.iter().all(|v| v.to_bits() == expected),
                "augmentation={augmentation:?}"
            );
        }
    }
}

#[test]
fn reference_augmentation_rounds_two_products_before_addition() {
    // signal = 0x3f7fbe79, noise = 0xbf7fbe78: separately rounded
    // products differ by exactly 2^-24. Fusing the signal product with
    // the rounded noise product would instead produce 0x337f7cee.
    let actual =
        augmented_reference_values(0.999, f32::from_bits(0x3f80_0001), Some(-999.0)).unwrap();
    assert!(actual.iter().all(|v| v.to_bits() == 0x3380_0000));
}

#[test]
fn reference_augmentation_checks_f64_range_and_noise_before_rounding() {
    for augmentation in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -f64::from_bits(1),
        f64::from_bits(1.0_f64.to_bits() + 1),
    ] {
        assert!(augmented_reference_values(augmentation, 1.0, Some(1.0)).is_err());
    }
    assert!(augmented_reference_values(f64::from_bits(1.0_f64.to_bits() - 1), 1.0, None).is_err());
    let unchanged = augmented_reference_values(1.0, f32::from_bits(0x3f80_0001), None).unwrap();
    assert!(unchanged.iter().all(|v| v.to_bits() == 0x3f80_0001));
}

#[test]
fn adapter_second_allocation_failure_refunds_and_wrong_host_is_rejected() {
    // Use an unchanged valid text-to-image configuration, selecting its request.
    let mut c = config(2);
    c.requests.remove(0);
    let t = transformer();
    c = PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        3,
        c.requests,
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::default(),
    )
    .unwrap();
    let host = account(1 << 20);
    let image = f32_tensor(
        &host,
        "value.sample",
        vec![1, 2, 1, 4, 4],
        std::iter::repeat_n(0.5, 32),
    );
    let audio = f32_tensor(
        &host,
        "value.audio",
        vec![1, 2, 2, 2],
        std::iter::repeat_n(0.5, 8),
    );
    let connector = f32_tensor(
        &host,
        "value.connector",
        vec![3, 4],
        std::iter::repeat_n(0.5, 12),
    );
    let keys = cache::fixture_keys(&c, &connector);
    let (basis, cos, sin) = supplemental(&c, &host);
    let owned_before = host.usage().unwrap().committed_bytes();
    let image_bytes = (c.image_rows().unwrap() * c.image_width().unwrap() * 4) as usize;
    let padding = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            (1 << 20) - owned_before - image_bytes,
        )
        .unwrap();
    let before = host.usage().unwrap().committed_bytes();
    assert!(PH3ImageDiTInputAdapter::new(&c, &host)
        .unwrap()
        .prepare(
            &[PH3ImageDiTSample {
                image: &image,
                audio: &audio
            }],
            &[None],
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys
            },
            basis.clone(),
            cos.clone(),
            sin.clone()
        )
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    let alien = account(1 << 20);
    assert!(PH3ImageDiTInputAdapter::new(&c, &alien)
        .unwrap()
        .prepare(
            &[PH3ImageDiTSample {
                image: &image,
                audio: &audio
            }],
            &[None],
            PH3ImageDiTTextInput::WithTextRefiner {
                connector: &connector,
                keys: &keys
            },
            basis,
            cos,
            sin
        )
        .is_err());
    drop(padding);
}
