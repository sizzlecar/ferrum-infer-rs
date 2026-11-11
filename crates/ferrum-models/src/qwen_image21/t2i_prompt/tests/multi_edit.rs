use super::*;

fn grids() -> [QwenImage21EditImageGrid; 2] {
    [
        QwenImage21EditImageGrid::new(4, 4, 2).unwrap(),
        QwenImage21EditImageGrid::new(4, 8, 2).unwrap(),
    ]
}

#[test]
fn official_numbered_template_keeps_single_bytes_and_accounts_for_all_placeholders() {
    let user = "cat";
    let legacy = format!(
        "<|im_start|>system\n{T2I_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(tokenizer::render_edit(user, 1), legacy);
    let second = " <image2><|vision_start|><|image_pad|><|vision_end|>";
    assert_eq!(
        tokenizer::render_edit(user, 2),
        legacy.replace("<|vision_end|>cat", &format!("<|vision_end|>{second}cat"))
    );
    let tokenizer = tokenizer();
    let prepared = tokenizer.prepare_edit(user, 64, 4, 4, 2).unwrap();
    let (source, _, _) = tokenizer_sources();
    let engine = tokenizers::Tokenizer::from_bytes(source).unwrap();
    let mut expected = engine.encode(legacy, true).unwrap().get_ids().to_vec();
    let image_id = engine.token_to_id("<|image_pad|>").unwrap();
    let start = expected.iter().position(|&id| id == image_id).unwrap();
    expected.splice(start..start + 1, [image_id; 4]);
    assert_eq!(prepared.prompts().input_ids(), expected);
    assert_eq!(
        prepared.image_slot_ranges()[0].image_slot_start(),
        (start - tokenizer.drop_prefix_tokens()) as u32
    );
    assert_eq!(prepared.total_image_slots(), 4);
    assert_eq!(
        tokenizer.maximum_sequence_length_multi(64, &[4]).unwrap(),
        tokenizer.maximum_sequence_length(64, 4).unwrap()
    );
    let two = tokenizer
        .prepare_edit_batch_multi(&[user], 64, &grids())
        .unwrap();
    assert!(
        two.prompts().sequence_length()
            <= tokenizer
                .maximum_sequence_length_multi(64, &[4, 8])
                .unwrap()
    );
    let ten = [grids()[0]; 10];
    let prepared_ten = tokenizer
        .prepare_edit_batch_multi(&[user], 64, &ten)
        .unwrap();
    assert_eq!(prepared_ten.image_slot_ranges().len(), 10);
    assert_eq!(prepared_ten.total_image_slots(), 40);
    assert!(
        prepared_ten.prompts().sequence_length()
            <= tokenizer
                .maximum_sequence_length_multi(64, &[4; 10])
                .unwrap()
    );
    assert!(tokenizer
        .prepare_edit_batch_multi(&[user], 64, &[])
        .is_err());
    assert!(tokenizer
        .prepare_edit_batch_multi(&[user], 64, &[grids()[0]; 11])
        .is_err());
    assert!(tokenizer.maximum_sequence_length_multi(64, &[]).is_err());
    assert!(tokenizer.maximum_sequence_length_multi(64, &[0]).is_err());
    assert!(tokenizer
        .maximum_sequence_length_multi(64, &[usize::MAX, 1])
        .is_err());
    assert!(tokenizer
        .prepare_edit_batch_multi(&["<|image_pad|>"], 64, &grids())
        .is_err());
}

fn outputs(host: &HostMemoryAccount) -> Vec<Vec<Vec<EditVisionPart>>> {
    use crate::qwen3_vl_vision::Qwen3VLVisionFamily as Vision;
    (0..2)
        .map(|sample| {
            grids()
                .iter()
                .enumerate()
                .map(|(reference, grid)| {
                    std::iter::once(Vision::OUTPUT_ID)
                        .chain(Vision::DEEPSTACK_IDS)
                        .enumerate()
                        .map(|(part, id)| {
                            let slots = grid.image_slots() as usize;
                            let mut bytes = host
                                .allocate_zeroed(HostMemoryClass::Output, slots * 8)
                                .unwrap();
                            for (element, b) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                                b.copy_from_slice(
                                    &(0x3f00_u16
                                        + sample * 512
                                        + reference as u16 * 256
                                        + part as u16 * 32
                                        + element as u16)
                                        .to_le_bytes(),
                                );
                            }
                            (
                                ProgramValueId::new(id).unwrap(),
                                ResolvedTensorSpec::new(
                                    vec![slots as u64, 4],
                                    ElementType::Bf16,
                                    ResolvedTensorLayout::Contiguous,
                                )
                                .unwrap(),
                                bytes.freeze(),
                            )
                        })
                        .collect()
                })
                .collect()
        })
        .collect()
}

#[test]
fn multi_reference_bridge_preserves_order_mrope_and_real_host_boundary() {
    let tokenizer = tokenizer();
    let prompt = tokenizer
        .prepare_edit_batch_multi(&["cat", "blue cat"], 64, &grids())
        .unwrap();
    assert_eq!(prompt.total_image_slots(), 12);
    assert_eq!(prompt.spans.len(), 4);
    let sequence = prompt.prompts().sequence_length();
    let a = architecture_with_layers(3);
    let source_bytes = 2 * 12 * 8 * 4;
    let input_bytes = 2 * sequence * 8 * 3 + 2 * 12 * 8 * 3;
    let host = account(source_bytes + input_bytes);
    let outputs = outputs(&host);
    let mut parts = outputs
        .iter()
        .map(|sample| edit_vision_parts(sample))
        .collect::<Vec<_>>();
    parts[1][1].reverse();
    let table = embedding_bytes();
    let occupied = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .unwrap();
    assert!(prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &parts)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), source_bytes + 1);
    drop(occupied);
    let inputs = prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &parts)
        .unwrap();
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        source_bytes + input_bytes
    );
    for (row, &token) in prompt.prompts().input_ids().iter().enumerate() {
        let matching = prompt.spans.iter().enumerate().find(|(_, span)| {
            let run = span.visual_run();
            (run.start()..run.start() + run.length()).contains(&(row as u64))
        });
        let expected = if let Some((index, span)) = matching {
            let local = row - span.visual_run().start() as usize;
            &outputs[index / 2][index % 2][0].2.bytes()[local * 8..(local + 1) * 8]
        } else {
            &table[token as usize * 8..(token as usize + 1) * 8]
        };
        assert_eq!(&inputs[0].bytes().bytes()[row * 8..(row + 1) * 8], expected);
    }
    for (level, input) in inputs[3..].iter().enumerate() {
        assert_eq!(input.tensor().dimensions(), [24, 4]);
        let expected = outputs
            .iter()
            .flatten()
            .flat_map(|reference| reference[level + 1].2.bytes().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(input.bytes().bytes(), expected);
        for reference in outputs.iter().flatten() {
            assert!(!input.bytes().same_region(&reference[level + 1].2));
        }
    }
    // Independent explicit position expectation: the first 2x2 grid advances
    // positions by 2 (not four slots); the second 2x4 grid advances by 4.
    for sample in 0..2 {
        let left = sequence - prompt.prompts().valid_input_lengths()[sample];
        for reference in 0..2 {
            let span = &prompt.spans[sample * 2 + reference];
            let start = span.visual_run().start() as usize;
            let origin = start - sample * sequence - left - if reference == 0 { 0 } else { 2 };
            assert_eq!(
                start - sample * sequence - left - tokenizer.drop_prefix_tokens(),
                prompt.image_slot_ranges()[reference].image_slot_start() as usize
            );
            for j in 0..span.visual_run().length() as usize {
                let t = origin as f32;
                let h = (origin + j / span.merged_width() as usize) as f32 * 0.1;
                let c = [
                    bf16::from_f32(libm::cosf(t)).to_bits(),
                    bf16::from_f32(libm::cosf(h)).to_bits(),
                ];
                let s = [
                    bf16::from_f32(libm::sinf(t)).to_bits(),
                    bf16::from_f32(libm::sinf(h)).to_bits(),
                ];
                assert_eq!(
                    bits(&inputs[1].bytes().bytes()[(start + j) * 8..(start + j + 1) * 8]),
                    [c[0], c[1], c[0], c[1]]
                );
                assert_eq!(
                    bits(&inputs[2].bytes().bytes()[(start + j) * 8..(start + j + 1) * 8]),
                    [s[0], s[1], s[0], s[1]]
                );
            }
        }
    }
    drop(parts);
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), input_bytes);
    drop(inputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn multi_reference_bridge_rejects_missing_or_reordered_geometry_without_leaking() {
    let prompt = tokenizer()
        .prepare_edit_batch_multi(&["cat", "blue cat"], 64, &grids())
        .unwrap();
    let host = account(16384);
    let outputs = outputs(&host);
    let parts = outputs
        .iter()
        .map(|sample| edit_vision_parts(sample))
        .collect::<Vec<_>>();
    let a = architecture_with_layers(3);
    let table = embedding_bytes();
    let before = host.usage().unwrap();
    let mut bad = parts.clone();
    bad[1].pop();
    assert!(prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &bad)
        .is_err());
    bad = parts.clone();
    bad[1].swap(0, 1);
    assert!(prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &bad)
        .is_err());
    bad = parts.clone();
    bad[1][1].pop();
    assert!(prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &bad)
        .is_err());
    bad = parts.clone();
    bad[1][1][3] = bad[1][1][2];
    assert!(prompt
        .managed_text_batch_parts_multi(&host, &a, &table, &bad)
        .is_err());
    // The legacy single-reference owner API must not silently accept one image
    // for a multi-reference prompt.
    assert!(prompt
        .managed_text_batch_parts(&host, &a, &table, &parts[0])
        .is_err());
    assert_eq!(host.usage().unwrap(), before);
}
