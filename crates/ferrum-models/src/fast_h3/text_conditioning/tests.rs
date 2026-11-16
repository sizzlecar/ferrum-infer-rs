use super::*;
use crate::qwen3_vl_vision::Qwen3VLVisionGrid;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
mod video;

fn setup() -> (Tokenizer, Qwen3VLVisionConfig, serde_json::Value) {
    let vocab = [
        ("<Picture", 0),
        ("1>:", 1),
        ("cat", 2),
        ("[UNK]", 3),
        ("<|vision_start|>", 4),
        ("<|image_pad|>", 5),
        ("<|vision_end|>", 6),
        ("2>:", 7),
    ]
    .into_iter()
    .map(|(t, id)| (t.to_owned(), id))
    .collect();
    let mut tokenizer = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    for name in ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"] {
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from(name, true)]);
    }
    let raw = serde_json::json!({"architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",
        "vision_config":{"model_type":"qwen3_vl","dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh","depth":3,"hidden_size":8,"intermediate_size":16,"num_heads":2,"num_position_embeddings":16,"out_hidden_size":128,"deepstack_visual_indexes":[0,1,2],"in_channels":3,"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2},
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","attention_bias":false,"attention_dropout":0.0,"hidden_act":"silu","num_hidden_layers":3,"hidden_size":128,"intermediate_size":256,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":128,"vocab_size":8,"max_position_embeddings":128,"rms_norm_eps":0.000001,"rope_theta":5000000,"rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[24,20,20]}}});
    let vision = Qwen3VLVisionConfig::from_hf_json(
        &serde_json::to_vec(&raw).unwrap(),
        vec![Qwen3VLVisionGrid::new(1, 2, 4).unwrap()],
    )
    .unwrap();
    (tokenizer, vision, raw)
}

#[test]
fn mixed_reference_labels_preserve_order_and_separate_image_audio_numbering() {
    let (mut tokenizer, image, _) = setup();
    tokenizer.add_tokens(&[tokenizers::AddedToken::from("<Audio", false)]);
    let audio_id = tokenizer.token_to_id("<Audio").unwrap();
    let references = [
        FastH3PresentationReference::Audio,
        FastH3PresentationReference::Image(&image),
        FastH3PresentationReference::Audio,
        FastH3PresentationReference::Image(&image),
    ];
    let limits = FastH3PresentationLimits {
        maximum_prompt_bytes: 128,
        maximum_tokens: 17,
    };
    let p = FastH3MediaPresentation::ordered_references(&tokenizer, "cat", &references, limits)
        .unwrap();
    assert_eq!(
        p.token_ids(),
        [audio_id, 1, 0, 1, 4, 5, 5, 6, audio_id, 7, 0, 7, 4, 5, 5, 6, 2]
    );
    assert_eq!(
        p.image_spans()
            .iter()
            .map(|s| (s.visual_run().start(), s.visual_run().length()))
            .collect::<Vec<_>>(),
        [(5, 2), (13, 2)]
    );
    assert_eq!(
        p.text_tags()
            .iter()
            .map(|r| (r.rows, r.tag))
            .collect::<Vec<_>>(),
        [
            (2, FastH3TextTokenTag::Text),
            (2, FastH3TextTokenTag::Text),
            (4, FastH3TextTokenTag::Vision),
            (2, FastH3TextTokenTag::Text),
            (2, FastH3TextTokenTag::Text),
            (4, FastH3TextTokenTag::Vision),
            (1, FastH3TextTokenTag::Text),
        ]
    );
    assert!(FastH3MediaPresentation::ordered_references(
        &tokenizer,
        "cat",
        &references,
        FastH3PresentationLimits {
            maximum_tokens: 16,
            ..limits
        }
    )
    .is_err());
    assert!(FastH3MediaPresentation::ordered_references(
        &tokenizer,
        "cat",
        &[FastH3PresentationReference::Audio],
        limits
    )
    .is_err());
}

#[test]
fn raw_picture_presentation_replaces_only_pad_rows_and_admits_three_original_updates() {
    let (tokenizer, vision, raw) = setup();
    let limits = FastH3PresentationLimits {
        maximum_prompt_bytes: 128,
        maximum_tokens: 32,
    };
    let p = FastH3ImagePresentation::single_image(&tokenizer, "cat", &vision, limits).unwrap();
    assert_eq!(p.token_ids(), [0, 1, 4, 5, 5, 6, 2]);
    assert_eq!(
        p.text_tags()
            .iter()
            .map(|r| (r.rows, r.tag))
            .collect::<Vec<_>>(),
        [
            (2, FastH3TextTokenTag::Text),
            (4, FastH3TextTokenTag::Vision),
            (1, FastH3TextTokenTag::Text)
        ]
    );
    assert_eq!(p.image_spans()[0].visual_run().start(), 3);
    let text =
        Qwen3VLTextConfig::from_hf_json(&serde_json::to_vec(&raw).unwrap(), p.layout().unwrap())
            .unwrap()
            .with_deepstack(p.deepstack().unwrap())
            .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let embedding = (0..8)
        .flat_map(|id| {
            bf16::from_f32(id as f32)
                .to_bits()
                .to_le_bytes()
                .repeat(128)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            Text::EMBEDDING_EXTERNAL_NAME,
            TensorView::new(Dtype::BF16, vec![8, 128], &embedding).unwrap(),
        )],
        &None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(dir.path()).unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let outputs = std::iter::once(Vision::OUTPUT_ID)
        .chain(Vision::DEEPSTACK_IDS)
        .enumerate()
        .map(|(i, name)| {
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::Output, 2 * 128 * 2)
                .unwrap();
            for b in bytes.as_mut().chunks_exact_mut(2) {
                b.copy_from_slice(&bf16::from_f32(20. + i as f32).to_bits().to_le_bytes());
            }
            (
                ProgramValueId::new(name).unwrap(),
                ResolvedTensorSpec::new(
                    vec![2, 128],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
        })
        .collect::<Vec<_>>();
    let views = std::array::from_fn(|i| {
        FastH3VisionFeatureView::new(&outputs[i].0, &outputs[i].1, &outputs[i].2)
    });
    let inputs = p
        .managed_text_inputs(&text, &archive, std::slice::from_ref(&views), &host)
        .unwrap();
    assert_eq!(inputs.len(), 6);
    let hidden = inputs
        .iter()
        .find(|i| i.value_id().as_str() == Text::EMBEDDINGS_ID)
        .unwrap();
    for (row, value) in [0., 1., 4., 20., 20., 6., 2.].iter().enumerate() {
        assert_eq!(
            &hidden.bytes().bytes()[row * 256..row * 256 + 2],
            &bf16::from_f32(*value).to_bits().to_le_bytes()
        );
    }
    for (i, name) in Text::DEEPSTACK_IDS.iter().enumerate() {
        let input = inputs
            .iter()
            .find(|v| v.value_id().as_str() == *name)
            .unwrap();
        assert_eq!(input.bytes().bytes(), outputs[i + 1].2.bytes());
        assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
        assert_ne!(
            input.bytes().bytes().as_ptr(),
            outputs[i + 1].2.bytes().as_ptr()
        );
    }
    assert!(
        FastH3ImagePresentation::single_image(&tokenizer, "<|image_pad|>", &vision, limits)
            .is_err()
    );
    assert!(FastH3ImagePresentation::single_image(
        &tokenizer,
        "cat",
        &vision,
        FastH3PresentationLimits {
            maximum_tokens: 6,
            ..limits
        }
    )
    .is_err());
    drop((inputs, outputs));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

fn feature_outputs(
    host: &HostMemoryAccount,
    rows: usize,
    base: f32,
) -> [(ProgramValueId, ResolvedTensorSpec, ManagedHostBytes); 4] {
    let names = [
        Vision::OUTPUT_ID,
        Vision::DEEPSTACK_IDS[0],
        Vision::DEEPSTACK_IDS[1],
        Vision::DEEPSTACK_IDS[2],
    ];
    std::array::from_fn(|i| {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::Output, rows * 128 * 2)
            .unwrap();
        for b in bytes.as_mut().chunks_exact_mut(2) {
            b.copy_from_slice(&bf16::from_f32(base + i as f32).to_bits().to_le_bytes());
        }
        (
            ProgramValueId::new(names[i]).unwrap(),
            ResolvedTensorSpec::new(
                vec![rows as u64, 128],
                ElementType::Bf16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            bytes.freeze(),
        )
    })
}

#[test]
fn ordered_images_route_each_spatial_span_and_deepstack_without_retaining_failed_claims() {
    let (tokenizer, first, raw) = setup();
    let second = Qwen3VLVisionConfig::from_hf_json(
        &serde_json::to_vec(&raw).unwrap(),
        vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap()],
    )
    .unwrap();
    let limits = FastH3PresentationLimits {
        maximum_prompt_bytes: 128,
        maximum_tokens: 12,
    };
    let p = FastH3ImagePresentation::ordered_images(&tokenizer, "cat", &[&first, &second], limits)
        .unwrap();
    assert_eq!(p.token_ids(), [0, 1, 4, 5, 5, 6, 0, 7, 4, 5, 6, 2]);
    assert_eq!(
        p.image_spans()
            .iter()
            .map(|s| (s.visual_run().start(), s.visual_run().length()))
            .collect::<Vec<_>>(),
        [(3, 2), (9, 1)]
    );
    assert_eq!(
        p.text_tags()
            .iter()
            .map(|r| (r.rows, r.tag))
            .collect::<Vec<_>>(),
        [
            (2, FastH3TextTokenTag::Text),
            (4, FastH3TextTokenTag::Vision),
            (2, FastH3TextTokenTag::Text),
            (3, FastH3TextTokenTag::Vision),
            (1, FastH3TextTokenTag::Text),
        ]
    );
    assert!(FastH3ImagePresentation::ordered_images(
        &tokenizer,
        "cat",
        &[&first, &second],
        FastH3PresentationLimits {
            maximum_tokens: 11,
            ..limits
        }
    )
    .is_err());
    assert!(FastH3ImagePresentation::ordered_images(&tokenizer, "cat", &[], limits).is_err());
    let text =
        Qwen3VLTextConfig::from_hf_json(&serde_json::to_vec(&raw).unwrap(), p.layout().unwrap())
            .unwrap()
            .with_deepstack(p.deepstack().unwrap())
            .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let embedding = (0..8)
        .flat_map(|id| {
            bf16::from_f32(id as f32)
                .to_bits()
                .to_le_bytes()
                .repeat(128)
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            Text::EMBEDDING_EXTERNAL_NAME,
            TensorView::new(Dtype::BF16, vec![8, 128], &embedding).unwrap(),
        )],
        &None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let archive = SafetensorsArchive::open(dir.path()).unwrap();
    let account = |bytes| {
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
    };
    // The smaller real account can hold both original outputs, but cannot
    // simultaneously admit the complete text input. Failed preparation keeps
    // those original outputs and releases every new reservation.
    for capacity in [6400, 1 << 20] {
        let host = account(capacity);
        let outputs = [
            feature_outputs(&host, 2, 20.0),
            feature_outputs(&host, 1, 40.0),
        ];
        let views: [[FastH3VisionFeatureView<'_>; 4]; 2] = std::array::from_fn(|reference| {
            std::array::from_fn(|i| {
                // Native output order need not match the semantic ID order.
                let output = &outputs[reference][[3, 0, 2, 1][i]];
                FastH3VisionFeatureView::new(&output.0, &output.1, &output.2)
            })
        });
        let before = host.usage().unwrap().committed_bytes();
        let footprint = p.host_footprint(text.architecture(), &host).unwrap();
        assert_eq!(footprint.vision_output_bytes, before);
        assert!(p
            .managed_text_inputs(&text, &archive, &views[..1], &host)
            .is_err());
        assert!(p
            .managed_text_inputs(&text, &archive, &[views[1], views[0]], &host)
            .is_err());
        let prepared = p.managed_text_inputs(&text, &archive, &views, &host);
        if capacity == 6400 {
            assert!(prepared.is_err());
        } else {
            let inputs = prepared.unwrap();
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::InputPreprocess),
                footprint.input_bytes
            );
            let hidden = inputs
                .iter()
                .find(|i| i.value_id().as_str() == Text::EMBEDDINGS_ID)
                .unwrap();
            for (row, value) in [0., 1., 4., 20., 20., 6., 0., 7., 4., 40., 6., 2.]
                .into_iter()
                .enumerate()
            {
                for bytes in hidden.bytes().bytes()[row * 256..(row + 1) * 256].chunks_exact(2) {
                    assert_eq!(bytes, bf16::from_f32(value).to_bits().to_le_bytes());
                }
            }
            for (layer, id) in Text::DEEPSTACK_IDS.iter().enumerate() {
                let input = inputs
                    .iter()
                    .find(|i| i.value_id().as_str() == *id)
                    .unwrap();
                assert_eq!(input.tensor().dimensions(), [3, 128]);
                for (row, base) in [21.0_f32, 21.0, 41.0].into_iter().enumerate() {
                    for bytes in input.bytes().bytes()[row * 256..(row + 1) * 256].chunks_exact(2) {
                        assert_eq!(
                            bytes,
                            bf16::from_f32(base + layer as f32).to_bits().to_le_bytes()
                        );
                    }
                }
            }
            drop(inputs);
            let foreign = account(1 << 20);
            let alien = feature_outputs(&foreign, 1, 40.0);
            let foreign_views = std::array::from_fn(|i| {
                FastH3VisionFeatureView::new(&alien[i].0, &alien[i].1, &alien[i].2)
            });
            assert!(p
                .managed_text_inputs(&text, &archive, &[views[0], foreign_views], &host)
                .is_err());
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        assert_eq!(host.usage().unwrap().reserved_bytes(), 0);
        drop(outputs);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}
