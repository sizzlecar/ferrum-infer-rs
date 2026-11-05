//! Original image presentation and managed F32 input boundaries. Numerical
//! execution tests use real Reference visual readbacks; no CUDA claim is made.
use super::*;
mod bridge;

pub(super) fn tokenizer(max_bytes: usize, max_tokens: u32) -> PH3ImageTextTokenizer {
    let vocabulary = [
        ("<Picture", 0),
        ("1>:", 1),
        ("cat", 2),
        ("after", 3),
        ("[UNK]", 4),
        ("<|vision_start|>", 11),
        ("<|vision_end|>", 12),
        ("<|image_pad|>", 13),
        ("<|video_pad|>", 14),
        ("<|endoftext|>", 15),
    ]
    .into_iter()
    .map(|(text, id)| (text.to_owned(), id))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut engine = tokenizers::Tokenizer::new(model);
    engine.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    engine.add_special_tokens(&[
        tokenizers::AddedToken::from("<|endoftext|>", true),
        tokenizers::AddedToken::from("<|vision_start|>", true),
        tokenizers::AddedToken::from("<|vision_end|>", true),
        tokenizers::AddedToken::from("<|image_pad|>", true),
        tokenizers::AddedToken::from("<|video_pad|>", true),
    ]);
    let source = engine.to_string(false).unwrap();
    PH3ImageTextTokenizer::from_source_bytes(
        source.as_bytes(),
        br#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"<|endoftext|>","bos_token":null}"#,
        16,
        PH3ImageTextTokenizerLimits {
            max_source_bytes: 1 << 20,
            max_prompt_bytes: max_bytes,
            max_tokens_per_prompt: max_tokens,
            max_batch: 1,
        },
    )
    .unwrap()
}
fn grid(height: u64, width: u64) -> Qwen3VLVisionGrid {
    Qwen3VLVisionGrid::new(1, height, width).unwrap()
}
fn architecture(sections: [u64; 3]) -> Qwen3VLTextArchitecture {
    Qwen3VLTextArchitecture::from_hf_value(&serde_json::json!({
        "architectures":["Qwen3VLForConditionalGeneration"],"model_type":"qwen3_vl",
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16",
            "attention_bias":false,"attention_dropout":0.0,"hidden_act":"silu",
            "num_hidden_layers":5,"hidden_size":256,"intermediate_size":256,
            "num_attention_heads":2,"num_key_value_heads":1,"head_dim":128,
            "vocab_size":16,"max_position_embeddings":128,"rms_norm_eps":0.25,
            "rope_theta":5000000,"rope_scaling":{"rope_type":"default",
                "mrope_interleaved":true,"mrope_section":sections}}
    }))
    .unwrap()
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn original_label_markers_and_expanded_slots_have_distinct_visual_and_modality_ranges() {
    let t = tokenizer(128, 128);
    let prompt = t.prepare_single_image("cat after", &grid(4, 6), 2).unwrap();
    assert_eq!(
        prompt.text().token_ids(),
        &[0, 1, 11, 15, 15, 15, 15, 15, 15, 12, 2, 3]
    );
    assert_eq!(prompt.text().valid_lengths(), &[12]);
    assert_eq!(prompt.text().layout().sequence_length(), 12);
    assert_eq!(prompt.visual_rows().rows().unwrap(), 6);
    let runs = prompt.visual_rows().runs();
    assert_eq!(runs.len(), 1);
    assert_eq!((runs[0].start(), runs[0].length()), (3, 6));
    assert_eq!(
        prompt.token_tags(),
        &[
            PH3ImageModality::Text,
            PH3ImageModality::Text,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Image,
            PH3ImageModality::Text,
            PH3ImageModality::Text,
        ]
    );
    // Both boundary tokens keep their original text embedding while the full
    // marker/feature block receives the DiT Image modality tag.
    assert_eq!(prompt.text().token_ids()[runs[0].start() as usize - 1], 11);
    assert_eq!(
        prompt.text().token_ids()[(runs[0].start() + runs[0].length()) as usize],
        12
    );
}

#[test]
fn empty_raw_prompt_adds_no_extra_pad_and_non_square_positions_resume_after_largest_axis() {
    let t = tokenizer(128, 128);
    let empty = t.prepare_single_image("", &grid(4, 6), 2).unwrap();
    assert_eq!(
        empty.text().token_ids(),
        &[0, 1, 11, 15, 15, 15, 15, 15, 15, 12]
    );
    assert_eq!(empty.text().valid_lengths(), &[10]);
    assert!(empty.text().layout().key_runs()[0]
        .iter()
        .all(Qwen3VLTextKeyRun::is_valid));
    assert_eq!(empty.text().token_ids().last(), Some(&12));
    let landscape = t.prepare_single_image("cat after", &grid(4, 6), 2).unwrap();
    let expected = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [3, 3, 4],
        [3, 3, 5],
        [3, 4, 3],
        [3, 4, 4],
        [3, 4, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
    ];
    for (row, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            landscape.position_ids(row).unwrap(),
            expected,
            "landscape row {row}"
        );
    }
    let portrait = t.prepare_single_image("cat after", &grid(6, 4), 2).unwrap();
    let expected = [
        [3, 3, 3],
        [3, 3, 4],
        [3, 4, 3],
        [3, 4, 4],
        [3, 5, 3],
        [3, 5, 4],
    ];
    for (local, expected) in expected.into_iter().enumerate() {
        assert_eq!(
            portrait.position_ids(3 + local).unwrap(),
            expected,
            "portrait cell {local}"
        );
    }
    assert_eq!(portrait.position_ids(10).unwrap(), [7; 3]);
    assert!(portrait.position_ids(12).is_err());
}

#[test]
fn interleaved_rotary_uses_independent_thw_axes_and_original_f32_frequency_precision() {
    let p = tokenizer(128, 128)
        .prepare_single_image("cat after", &grid(4, 6), 2)
        .unwrap();
    let coordinates = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [3, 3, 4],
        [3, 3, 5],
        [3, 4, 3],
        [3, 4, 4],
        [3, 4, 5],
        [6, 6, 6],
        [7, 7, 7],
        [8, 8, 8],
    ];
    for sections in [[16, 24, 24], [62, 1, 1]] {
        let a = architecture(sections);
        let mut cos = vec![0u8; coordinates.len() * 128 * 4];
        let mut sin = cos.clone();
        p.write_rotary(&a, true, &mut cos).unwrap();
        p.write_rotary(&a, false, &mut sin).unwrap();
        let cos = floats(&cos);
        let sin = floats(&sin);
        // Explicit independent axis tables cover both the release's T/H/W
        // interleave and sections ending before the next H/W frequency.
        let mut axes = [0usize; 64];
        if sections == [16, 24, 24] {
            for triple in axes.chunks_exact_mut(3) {
                triple.copy_from_slice(&[0, 1, 2]);
            }
        } else {
            axes[1] = 1;
            axes[2] = 2;
        }
        for (row, pos) in coordinates.iter().enumerate() {
            for (pair, &axis) in axes.iter().enumerate() {
                let frequency = 5_000_000_f64.powf(-((2 * pair) as f64) / 128.0);
                let phase = f64::from(pos[axis]) * frequency;
                for (values, expected) in [(&cos, phase.cos()), (&sin, phase.sin())] {
                    let first = values[row * 128 + pair];
                    assert!((f64::from(first)-expected).abs() < 3e-6,
                        "sections {sections:?}, row {row}, frequency {pair}, axis {axis}: {first} vs {expected}");
                    assert_eq!(first.to_bits(), values[row * 128 + pair + 64].to_bits());
                }
            }
        }
        assert!(cos.iter().chain(&sin).any(|v| v.to_bits() & 0xffff != 0));
    }
    let a = architecture([16, 24, 24]);
    let mut short = vec![0xa5; coordinates.len() * 128 * 4 - 1];
    assert!(p.write_rotary(&a, true, &mut short).is_err());
    assert!(short.iter().all(|&b| b == 0xa5));
    let mut retry = vec![0; coordinates.len() * 128 * 4];
    p.write_rotary(&a, true, &mut retry).unwrap();
    assert!(floats(&retry)[..128].iter().all(|&x| x == 1.0));
}

#[test]
fn expansion_and_raw_text_limits_reject_without_truncating_or_consuming_a_prompt() {
    let exact = tokenizer(7, 12);
    let g = grid(4, 6);
    assert_eq!(
        exact
            .prepare_single_image("cat cat", &g, 2)
            .unwrap()
            .text()
            .token_ids()
            .len(),
        12
    );
    assert!(tokenizer(6, 12)
        .prepare_single_image("cat cat", &g, 2)
        .is_err());
    assert!(tokenizer(128, 11)
        .prepare_single_image("cat after", &g, 2)
        .is_err());
    assert_eq!(
        tokenizer(128, 11)
            .prepare_single_image("cat", &g, 2)
            .unwrap()
            .text()
            .valid_lengths(),
        &[11]
    );
    let generous = tokenizer(128, 128);
    for prompt in [
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ] {
        let error = generous.prepare_single_image(prompt, &g, 2).err().unwrap();
        assert!(
            error.to_string().contains("another vision block"),
            "{prompt}: {error}"
        );
    }
    assert!(generous
        .prepare_single_image("embedding:custom", &g, 2)
        .is_err());
    for (g, merge) in [
        (grid(4, 6), 0),
        (grid(4, 6), 4),
        (grid(3, 6), 2),
        (
            serde_json::from_value(serde_json::json!({"temporal":2,"height":4,"width":6})).unwrap(),
            2,
        ),
    ] {
        assert!(exact.prepare_single_image("cat", &g, merge).is_err());
    }
    assert_eq!(
        exact
            .prepare_single_image("cat", &grid(4, 6), 2)
            .unwrap()
            .text()
            .token_ids()[10],
        2
    );
}
