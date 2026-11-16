use super::*;

#[test]
fn timestamped_video_splits_original_feature_rows_without_splitting_their_owners() {
    let (mut tokenizer, image, mut raw) = setup();
    tokenizer.add_special_tokens(&[tokenizers::AddedToken::from("<|video_pad|>", true)]);
    for token in ["<Video", "<Audio", "<0.2", "<1.0", "seconds>"] {
        tokenizer.add_tokens(&[tokenizers::AddedToken::from(token, false)]);
    }
    let vocabulary = tokenizer.get_vocab_size(true);
    raw["text_config"]["vocab_size"] = vocabulary.into();
    let video = Qwen3VLVisionConfig::new(
        image.architecture().clone(),
        vec![Qwen3VLVisionGrid::new(2, 2, 4).unwrap()],
    )
    .unwrap();
    let references = [
        FastH3PresentationReference::Video {
            vision: &video,
            block_timestamps: &[0.25, 1.0],
            has_audio: true,
        },
        FastH3PresentationReference::Image(&image),
        FastH3PresentationReference::Audio,
    ];
    let limits = FastH3PresentationLimits {
        maximum_prompt_bytes: 128,
        maximum_tokens: 64,
    };
    let p = FastH3MediaPresentation::ordered_references(&tokenizer, "cat", &references, limits)
        .unwrap();
    let raw_ids = |s| tokenizer.encode(s, false).unwrap().get_ids().to_vec();
    let video_pad = tokenizer.token_to_id("<|video_pad|>").unwrap();
    let mut expected = Vec::new();
    for s in ["<Audio 1>: ", "<Video 1>: ", "<0.2 seconds>"] {
        expected.extend(raw_ids(s));
    }
    expected.extend([4, video_pad, video_pad, 6]);
    expected.extend(raw_ids("<1.0 seconds>"));
    expected.extend([4, video_pad, video_pad, 6]);
    expected.extend(raw_ids("<Picture 1>: "));
    expected.extend([4, 5, 5, 6]);
    expected.extend(raw_ids("<Audio 2>: "));
    expected.extend(raw_ids("cat"));
    assert_eq!(p.token_ids(), expected);
    assert_eq!(p.image_spans().len(), 3);
    assert_eq!(
        p.sources
            .iter()
            .map(|s| (s.span_count, s.rows))
            .collect::<Vec<_>>(),
        [(2, 4), (1, 2)]
    );
    let text =
        Qwen3VLTextConfig::from_hf_json(&serde_json::to_vec(&raw).unwrap(), p.layout().unwrap())
            .unwrap()
            .with_deepstack(p.deepstack().unwrap())
            .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let embedding = (0..vocabulary)
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
            TensorView::new(Dtype::BF16, vec![vocabulary, 128], &embedding).unwrap(),
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
    let names = [
        Vision::OUTPUT_ID,
        Vision::DEEPSTACK_IDS[0],
        Vision::DEEPSTACK_IDS[1],
        Vision::DEEPSTACK_IDS[2],
    ];
    let outputs = [(4, 20_f32), (2, 40.)].map(|(rows, base)| {
        names.map(|name| {
            let layer = names.iter().position(|n| *n == name).unwrap();
            let mut bytes = host
                .allocate_zeroed(HostMemoryClass::Output, rows * 256)
                .unwrap();
            for (row, cells) in bytes.as_mut().chunks_exact_mut(256).enumerate() {
                for cell in cells.chunks_exact_mut(2) {
                    cell.copy_from_slice(
                        &bf16::from_f32(base + row as f32 + layer as f32 * 4.)
                            .to_bits()
                            .to_le_bytes(),
                    );
                }
            }
            (
                ProgramValueId::new(name).unwrap(),
                ResolvedTensorSpec::new(
                    vec![rows as u64, 128],
                    ElementType::Bf16,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
        })
    });
    let views: [[FastH3VisionFeatureView<'_>; 4]; 2] = std::array::from_fn(|s| {
        std::array::from_fn(|i| {
            FastH3VisionFeatureView::new(&outputs[s][i].0, &outputs[s][i].1, &outputs[s][i].2)
        })
    });
    let before = host.usage().unwrap().committed_bytes();
    let footprint = p.host_footprint(text.architecture(), &host).unwrap();
    assert_eq!(footprint.vision_output_bytes, before);
    let inputs = p
        .managed_text_inputs(&text, &archive, &views, &host)
        .unwrap();
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        footprint.input_bytes
    );
    let hidden = inputs
        .iter()
        .find(|t| t.value_id().as_str() == Text::EMBEDDINGS_ID)
        .unwrap();
    let expected_visual = [20., 21., 22., 23., 40., 41.];
    let mut cursor = 0;
    for (row, token) in p.token_ids().iter().enumerate() {
        let value = if [5, video_pad].contains(token) {
            let value = expected_visual[cursor];
            cursor += 1;
            value
        } else {
            *token as f32
        };
        for cell in hidden.bytes().bytes()[row * 256..(row + 1) * 256].chunks_exact(2) {
            assert_eq!(cell, bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    for (layer, id) in Text::DEEPSTACK_IDS.iter().enumerate() {
        let updates = inputs
            .iter()
            .find(|t| t.value_id().as_str() == *id)
            .unwrap();
        for (row, value) in expected_visual.iter().enumerate() {
            assert_eq!(
                &updates.bytes().bytes()[row * 256..row * 256 + 2],
                &bf16::from_f32(value + (layer + 1) as f32 * 4.)
                    .to_bits()
                    .to_le_bytes()
            );
        }
    }
    // Video blocks keep independent T1 spatial mRoPE, as in the original
    // expansion of video_grid_thw before the text encoder's position builder.
    assert_eq!(
        p.image_spans()
            .iter()
            .map(|s| (s.merged_height(), s.merged_width()))
            .collect::<Vec<_>>(),
        [(1, 2); 3]
    );
    for bad in [&[0.25][..], &[0.25, f64::NAN][..], &[1., 0.25][..]] {
        assert!(FastH3MediaPresentation::ordered_references(
            &tokenizer,
            "cat",
            &[FastH3PresentationReference::Video {
                vision: &video,
                block_timestamps: bad,
                has_audio: false
            }],
            limits
        )
        .is_err());
    }
    assert!(FastH3MediaPresentation::ordered_references(
        &tokenizer,
        "<|video_pad|>",
        &references,
        limits
    )
    .is_err());
    assert!(FastH3MediaPresentation::ordered_references(
        &tokenizer,
        "cat",
        &references,
        FastH3PresentationLimits {
            maximum_tokens: p.token_ids().len() as u32 - 1,
            ..limits
        }
    )
    .is_err());
    drop(inputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), before);
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
