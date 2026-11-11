use super::*;

fn reference(start: u32, slots: u32, height: u32, width: u32) -> QwenImage21ReferenceImage {
    QwenImage21ReferenceImage::new(start, slots, height, width).unwrap()
}

fn edited() -> QwenImage21TransformerConfig {
    QwenImage21TransformerConfig::new(
        transformer(2, true, false),
        QwenImage21T2iLayout::new(10, 2, 2, vec![10, 8]).unwrap(),
    )
    .unwrap()
    .with_reference_images(vec![reference(1, 2, 2, 4), reference(5, 1, 2, 2)])
    .unwrap()
}

#[test]
fn multiple_images_keep_independent_rotary_frames_and_attention_blocks() {
    let config = edited();
    let l = config.layout();
    assert!(l.reference_image().is_none());
    assert_eq!(l.reference_latent_tokens(), 12);
    assert_eq!(l.target_start(), 19);
    assert_eq!(l.sequence_length(), 23);
    // text[0], image[1..9], text[9..11], image[11..15],
    // text[15..19] (last two padded for sample 1), target[19..23].
    for (row, position) in [
        (0, [0, 0, 0]),
        (1, [1, -1, -2]),
        (8, [1, 0, 1]),
        (9, [5, 5, 5]),
        (10, [6, 6, 6]),
        (11, [7, -1, -1]),
        (14, [7, 0, 0]),
        (15, [9, 9, 9]),
        (18, [12, 12, 12]),
        (19, [13, -1, -1]),
        (22, [13, 0, 0]),
    ] {
        assert_eq!(l.rotary_position(row), Some(position));
    }
    assert_eq!(l.rotary_position(23), None);
    let attention = QwenImage21AttentionConfig::new(
        config.transformer().clone(),
        l.attention_layout().unwrap(),
    )
    .unwrap();
    let q = ResolvedTensorSpec::new(
        vec![46, 16],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let geometry =
        StructuredAttentionBf16Geometry::new(&q, &q, &q, &q, &attention.attributes().unwrap())
            .unwrap();
    assert_eq!(geometry.mask().visible_end(1), Some(9));
    assert_eq!(geometry.mask().visible_end(9), Some(10));
    assert_eq!(geometry.mask().visible_end(11), Some(15));
    assert_eq!(geometry.mask().visible_end(19), Some(23));
    for row in 0..23 {
        assert_eq!(
            geometry.mask().key_is_valid(1, row),
            Some(!(17..19).contains(&row))
        );
    }
    let adjacent = l
        .clone()
        .with_reference_images(vec![reference(1, 2, 2, 4), reference(3, 1, 2, 2)])
        .unwrap();
    let attrs = QwenImage21AttentionConfig::new(
        config.transformer().clone(),
        adjacent.attention_layout().unwrap(),
    )
    .unwrap()
    .attributes()
    .unwrap();
    let adjacent_geometry = StructuredAttentionBf16Geometry::new(&q, &q, &q, &q, &attrs).unwrap();
    assert_eq!(adjacent_geometry.mask().visible_end(1), Some(9));
    assert_eq!(adjacent_geometry.mask().visible_end(9), Some(13));
    assert_eq!(adjacent.rotary_position(9), Some([5, -1, -1]));
}

#[test]
fn multiple_images_gather_in_sample_order_and_cache_the_entire_prefix() {
    let config = edited();
    let f = F::new().unwrap();
    for phase in [
        QwenImage21TransformerPhase::Full,
        QwenImage21TransformerPhase::PrefixPrefill,
        QwenImage21TransformerPhase::PrefixDecode,
    ] {
        let c = config.clone().with_execution_phase(phase).unwrap();
        let profiles = f.numerical_profiles(&c).unwrap();
        let p = f
            .semantic_program(
                &c,
                profiles
                    .resolve(&profile_id(F::REFERENCE_PROFILE_ID).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let nodes = &p.blocks()[0].nodes;
        let node = |name: &str| {
            nodes
                .iter()
                .find(|n| n.id.as_str() == format!("node.qwen_image21.transformer.{name}"))
        };
        if phase == QwenImage21TransformerPhase::PrefixDecode {
            assert!(node("interleaved_joint").is_none());
            assert!(!p
                .input_tensor_specs()
                .contains_key(&id(F::REFERENCE_LATENTS_ID)));
            assert_eq!(p.states().len(), 4);
        } else {
            let attrs = &node("interleaved_joint").unwrap().attributes;
            let list = |name: &str| match &attrs[&AttributeId::new(name).unwrap()] {
                SemanticValue::Integers(v) => v.clone(),
                _ => panic!("integer row map"),
            };
            for batch in 0..2 {
                let selected = list("run_lengths")
                    .into_iter()
                    .zip(list("source_starts"))
                    .zip(list("source_batch_strides"))
                    .flat_map(|((len, start), stride)| {
                        (0..len).map(move |i| batch * stride + start + i)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    selected,
                    [
                        0, 10, 11, 12, 13, 14, 15, 16, 17, 3, 4, 18, 19, 20, 21, 6, 7, 8, 9, 22,
                        23, 24, 25
                    ]
                    .map(|i| i + batch * 26)
                );
            }
            assert_eq!(
                p.input_tensor_specs()[&id(F::REFERENCE_LATENTS_ID)].dimensions,
                [24, 3]
            );
            assert_eq!(
                node("target_hidden").unwrap().attributes
                    [&AttributeId::new("source_starts").unwrap()],
                SemanticValue::Integers(vec![19])
            );
        }
        assert_eq!(
            p.input_tensor_specs()[&id(F::LATENTS_ID)].dimensions,
            [8, 3]
        );
        for port in c.prefix_cache_ports().unwrap() {
            assert_eq!(port.tensor().dimensions, [38, 16]);
        }
        c.compile_options().unwrap();
    }
}

#[test]
fn reference_wire_retains_single_image_compatibility_and_rejects_invalid_lists() {
    let config = edited();
    let wire = serde_json::to_value(&config).unwrap();
    assert!(wire["layout"].get("reference_image").is_none());
    assert_eq!(
        wire["layout"]["reference_images"].as_array().unwrap().len(),
        2
    );
    assert_eq!(
        serde_json::from_value::<QwenImage21TransformerConfig>(wire.clone()).unwrap(),
        config
    );
    let mut conflict = wire;
    conflict["layout"]["reference_image"] = serde_json::to_value(reference(1, 2, 2, 4)).unwrap();
    assert!(serde_json::from_value::<QwenImage21TransformerConfig>(conflict).is_err());
    let single = config.clone().with_reference_image(1, 2, 2, 4).unwrap();
    assert_eq!(single.layout().reference_images(), &[reference(1, 2, 2, 4)]);
    assert!(single.layout().reference_image().is_some());
    assert_eq!(
        serde_json::to_value(single.layout()).unwrap(),
        json!({
            "text_tokens":10,"latent_height":2,"latent_width":2,"valid_text_lengths":[10,8],
            "reference_image":{"image_slot_start":1,"image_slots":2,"latent_height":2,"latent_width":4}
        })
    );
    let l = config.layout();
    for references in [
        vec![reference(5, 1, 2, 2), reference(1, 2, 2, 4)],
        vec![reference(1, 2, 2, 4), reference(2, 1, 2, 2)],
        vec![reference(1, 2, 2, 4), reference(8, 1, 2, 2)],
        vec![reference(1, 2, 2, 4), reference(9, 1, 2, 2)],
    ] {
        assert!(l.clone().with_reference_images(references).is_err());
    }
    // Ten images exercise the real segment/run bounds; an eleventh exceeds
    // the model capability. Adjacent images remain distinct blocks.
    let l = QwenImage21T2iLayout::new(13, 2, 2, vec![13]).unwrap();
    let references = (1..=10).map(|i| reference(i, 1, 2, 2)).collect::<Vec<_>>();
    assert!(l.clone().with_reference_images(references.clone()).is_ok());
    let mut too_many = references;
    too_many.push(reference(11, 1, 2, 2));
    assert!(l.with_reference_images(too_many).is_err());
    // Very long text plus image frame increments may exceed the finite RoPE
    // domain even when the raw VLM sequence itself is valid.
    assert!(QwenImage21T2iLayout::new(8191, 2, 2, vec![8191])
        .unwrap()
        .with_reference_images(vec![reference(1, 1, 2, 2), reference(3, 1, 2, 2)])
        .is_err());
}
