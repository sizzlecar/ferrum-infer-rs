//! Independent small oracles for Comfy's single-image reference protocol.
//! `comfy/ldm/minimax/model.py` at 56f179f6c50dad0c4c860ae4da427415c8e86415:
//! patchify_video (42-49), PackedLayout (355-478), _embed_and_pack (560-604),
//! and modality/time routing (683-741). These tests need no weights or GPU and
//! do not establish Torch arithmetic parity or official-model image quality.
use super::*;

fn mixed_config(reverse: bool) -> PH3ImageDiTConfig {
    use PH3ImageModality::{Image, Text};
    let t = transformer();
    let target = PH3ImageReferenceImage::new(4, 4).unwrap();
    let reference = Some(PH3ImageReferenceImage::new(2, 4).unwrap());
    // Caller-supplied time rows are deliberately distinct for audio. The
    // engine's separate tests own calculation of max(image_time, augmentation).
    let edit = PH3ImageDiTRequest::new(
        &t,
        &[Text, Image, Text],
        target,
        reference,
        &[0, 0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
        1 << 20,
    )
    .unwrap();
    let generate = PH3ImageDiTRequest::new(
        &t,
        &[Image, Text, Image],
        target,
        None,
        &[3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3],
        1 << 20,
    )
    .unwrap();
    let requests = if reverse {
        vec![generate, edit]
    } else {
        vec![edit, generate]
    };
    PH3ImageDiTConfig::new(
        &t,
        PH3ImageCurveTimeEmbedding::new(4097, 16).unwrap(),
        5,
        requests,
        PH3ImageDiTPhase::WithTextRefiner,
        PH3ImageStackProjectionStorage::default(),
    )
    .unwrap()
}

#[test]
fn every_reference_patch_uses_its_own_patch_order_noise_after_request_reordering() {
    // For [1,2,1,2,4], row order is (y/2,x/2), column order (c,dy,dx).
    // These complete source-index lists are written independently of the
    // adapter/layout helpers; the second patch must not reuse the first noise.
    const REFERENCE: [usize; 16] = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15];
    const TARGET: [usize; 32] = [
        0, 1, 4, 5, 16, 17, 20, 21, 2, 3, 6, 7, 18, 19, 22, 23, 8, 9, 12, 13, 24, 25, 28, 29, 10,
        11, 14, 15, 26, 27, 30, 31,
    ];
    const AUDIO: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];
    for reverse in [false, true] {
        let host = account(1 << 20);
        {
            let c = mixed_config(reverse);
            let images = [0, 1].map(|request| {
                f32_tensor(
                    &host,
                    "value.oracle.target",
                    vec![1, 2, 1, 4, 4],
                    (0..32).map(|i| 1.00390625 + i as f32 / 32. + request as f32),
                )
            });
            let audio = [0, 1].map(|request| {
                f32_tensor(
                    &host,
                    "value.oracle.audio",
                    vec![1, 2, 2, 2],
                    (0..8).map(|i| 0.50390625 + i as f32 / 16. + request as f32),
                )
            });
            let reference = f32_tensor(
                &host,
                "value.oracle.reference",
                vec![1, 2, 1, 2, 4],
                (0..16).map(|i| 1.00390625 + i as f32 / 32.),
            );
            let noise = f32_tensor(
                &host,
                "value.oracle.patch_noise",
                vec![2, 8],
                (0..16).map(|i| (3 * i + 1) as f32 / 32. - 0.75),
            );
            let connector = f32_tensor(
                &host,
                "value.oracle.connector",
                vec![6, 4],
                (0..24).map(|i| i as f32 / 32.),
            );
            let originals = [
                &reference, &noise, &images[0], &images[1], &audio[0], &audio[1],
            ];
            let before: Vec<_> = originals
                .iter()
                .map(|t| t.bytes().bytes().to_vec())
                .collect();
            let order = if reverse { [1, 0] } else { [0, 1] };
            let samples = order.map(|i| PH3ImageDiTSample {
                image: &images[i],
                audio: &audio[i],
            });
            let references = order.map(|i| {
                (i == 0).then_some(PH3ImageDiTReference {
                    latent: &reference,
                    patch_noise: Some(&noise),
                    augmentation: 0.75,
                })
            });
            let keys = cache::fixture_keys(&c, &connector);
            let (basis, cos, sin) = supplemental(&c, &host);
            let prepared = PH3ImageDiTInputAdapter::new(&c, &host)
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
            let mut expected_image = Vec::new();
            let mut expected_audio = Vec::new();
            for request in order {
                if request == 0 {
                    for (patch_element, source) in REFERENCE.into_iter().enumerate() {
                        let signal = 0.75_f32 * (1.00390625 + source as f32 / 32.);
                        let noise = 0.25_f32 * ((3 * patch_element + 1) as f32 / 32. - 0.75);
                        expected_image.push((signal + noise).to_bits());
                    }
                }
                for source in TARGET {
                    expected_image.push(
                        bf16::from_f32(1.00390625 + source as f32 / 32. + request as f32)
                            .to_f32()
                            .to_bits(),
                    );
                }
                for source in AUDIO {
                    expected_audio.push(
                        bf16::from_f32(0.50390625 + source as f32 / 16. + request as f32)
                            .to_f32()
                            .to_bits(),
                    );
                }
            }
            for (id, dimensions, expected) in [
                (PH3ImageDiTFamily::IMAGE_PATCHES_ID, [10, 8], expected_image),
                (PH3ImageDiTFamily::AUDIO_PATCHES_ID, [8, 2], expected_audio),
            ] {
                let input = prepared
                    .inputs
                    .iter()
                    .find(|t| t.value_id().as_str() == id)
                    .unwrap();
                assert_eq!(input.tensor().dimensions(), dimensions);
                assert_eq!(input.tensor().element_type(), ElementType::F32);
                let actual: Vec<_> = input
                    .bytes()
                    .bytes()
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(actual, expected, "{id}, reverse={reverse}");
                assert!(host.owns(input.bytes()));
            }
            for (original, before) in originals.iter().zip(&before) {
                assert_eq!(original.bytes().bytes(), before);
            }
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

fn integers<'a>(node: &'a ProgramNode, attribute: &str) -> &'a [i64] {
    let SemanticValue::Integers(values) = &node.attributes[&AttributeId::new(attribute).unwrap()]
    else {
        panic!("missing integer attribute {attribute}")
    };
    values
}

#[test]
fn edit_prefix_routes_keep_full_attention_and_only_target_velocity_ranges() {
    for reverse in [false, true] {
        let c = mixed_config(reverse);
        let expected_runs = if reverse {
            vec![(0, 3), (16, 4), (6, 4), (3, 3), (10, 2), (20, 4), (12, 4)]
        } else {
            vec![(0, 3), (6, 2), (16, 4), (8, 4), (3, 3), (20, 4), (12, 4)]
        };
        assert_eq!(program::packing_runs(&c).unwrap(), expected_runs);
        // Explicit tables in packed order. Vision-prefix Image rows use their
        // request's target time; only the two actual reference patches use 2.
        let edit = [
            (0, 1),
            (0, 0),
            (0, 1),
            (2, 0),
            (2, 0),
            (1, 2),
            (1, 2),
            (1, 2),
            (1, 2),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
        ];
        let generate = [
            (3, 0),
            (3, 1),
            (3, 0),
            (4, 2),
            (4, 2),
            (4, 2),
            (4, 2),
            (3, 0),
            (3, 0),
            (3, 0),
            (3, 0),
        ];
        let expected_rows = if reverse {
            [&generate[..], &edit[..]]
        } else {
            [&edit[..], &generate[..]]
        };
        for (request, expected) in c.requests.iter().zip(expected_rows) {
            let actual: Vec<_> = request
                .block
                .runs()
                .iter()
                .flat_map(|run| {
                    std::iter::repeat_n(
                        (run.time_index(), run.modality() as u32),
                        run.length() as usize,
                    )
                })
                .collect();
            assert_eq!(actual, expected);
        }
        let (lengths, parameters, segments, image_starts, audio_starts, image_times, audio_times) =
            if reverse {
                (
                    vec![1, 1, 1, 4, 4, 1, 1, 1, 2, 4, 4],
                    vec![9, 10, 9, 14, 9, 1, 0, 1, 6, 5, 0],
                    vec![11, 13],
                    vec![7, 20],
                    vec![3, 16],
                    vec![3, 0],
                    vec![4, 1],
                )
            } else {
                (
                    vec![1, 1, 1, 2, 4, 4, 1, 1, 1, 4, 4],
                    vec![1, 0, 1, 6, 5, 0, 9, 10, 9, 14, 9],
                    vec![13, 11],
                    vec![9, 20],
                    vec![5, 16],
                    vec![0, 3],
                    vec![1, 4],
                )
            };
        for policy in [
            PH3ImageDenoiserAttentionPolicy::Sequential,
            PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1,
        ] {
            let body =
                program::compose(&c.clone().with_attention_policy(policy).unwrap(), limits())
                    .unwrap();
            let nodes = &body.program.blocks()[0].nodes;
            let pack = nodes
                .iter()
                .find(|n| n.id.as_str().ends_with("dit_forward.pack0"))
                .unwrap();
            assert_eq!(
                integers(pack, "source_starts"),
                expected_runs.iter().map(|r| r.0 as i64).collect::<Vec<_>>()
            );
            assert_eq!(
                integers(pack, "run_lengths"),
                expected_runs.iter().map(|r| r.1 as i64).collect::<Vec<_>>()
            );
            let main: Vec<_> = nodes
                .iter()
                .filter(|n| n.id.as_str().contains("curve_stack.layer"))
                .collect();
            let attention: Vec<_> = main
                .iter()
                .filter(|n| n.operation_id.as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID)
                .collect();
            assert!(!attention.is_empty());
            for node in attention {
                assert_eq!(integers(node, "segment_lengths"), segments);
            }
            let modulation: Vec<_> = main
                .iter()
                .filter(|n| {
                    matches!(
                        n.operation_id.as_str(),
                        ROW_SELECTED_MODULATION_BF16_OPERATION_ID
                            | ROW_SELECTED_ADDCMUL_BF16_OPERATION_ID
                    )
                })
                .collect();
            assert!(!modulation.is_empty());
            for node in modulation {
                assert_eq!(integers(node, "run_lengths"), lengths);
                assert_eq!(integers(node, "parameter_rows"), parameters);
            }
            for (stream, starts, times, output, shape) in [
                (
                    "image",
                    &image_starts,
                    &image_times,
                    PH3ImageDiTFamily::IMAGE_VELOCITY_ID,
                    [8, 8],
                ),
                (
                    "audio",
                    &audio_starts,
                    &audio_times,
                    PH3ImageDiTFamily::AUDIO_VELOCITY_ID,
                    [8, 2],
                ),
            ] {
                let suffix = format!("final_layer.{stream}.chunk0.gather");
                let gather = nodes
                    .iter()
                    .find(|n| n.id.as_str().ends_with(&suffix))
                    .unwrap();
                assert_eq!(integers(gather, "source_starts"), starts);
                assert_eq!(integers(gather, "run_lengths"), [4, 4]);
                let suffix = format!("final_layer.{stream}.chunk0.modulate");
                let modulate = nodes
                    .iter()
                    .find(|n| n.id.as_str().ends_with(&suffix))
                    .unwrap();
                assert_eq!(integers(modulate, "parameter_rows"), times);
                assert_eq!(integers(modulate, "run_lengths"), [4, 4]);
                let spec = &body.options.tensor_specs()[&ProgramValueId::new(output).unwrap()];
                assert_eq!(spec.dimensions, shape);
                assert_eq!(spec.element_type, ElementType::Bf16);
            }
        }
    }
}
