use super::super::tests::{limits, protocol, source, CONFIG, TEMPLATE};
use super::*;

fn grid(height: u64, width: u64) -> Qwen3VLVisionGrid {
    Qwen3VLVisionGrid::new(1, height, width).unwrap()
}
fn render(p: &PH3ImagePromptExpansionProtocol, count: u8) -> PH3ImagePromptExpansionInput {
    p.render(
        &PH3ImagePromptExpansionRequest::edit("make the sky blue", count)
            .unwrap()
            .with_max_new_tokens(2)
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn actual_template_and_tokenizer_expand_only_real_image_rows_and_keep_boundary_embeddings() {
    let (p, _) = protocol(&[]);
    let input = render(&p, 1);
    let result = p.encode_image_input(&input, &[grid(4, 6)], 2).unwrap();
    let run = &result.image_runs()[0];
    assert_eq!(
        (run.rows(), run.merged_height(), run.merged_width()),
        (6, 2, 3)
    );
    let expected = p
        .tokenizer
        .encode(input.rendered().replace(IMAGE, &IMAGE.repeat(6)), false)
        .unwrap();
    assert_eq!(result.tokens(), expected.get_ids());
    assert_eq!(result.image_row(run.token_start() - 1), None);
    for row in 0..6 {
        assert_eq!(result.image_row(run.token_start() + row), Some((0, row)));
    }
    assert_eq!(result.image_row(run.token_start() + 6), None);
    assert_eq!(result.image_row(result.tokens().len()), None);
    assert_eq!(result.rope_delta(), -3);
    assert_eq!(
        result.position(result.tokens().len()).unwrap(),
        [(result.tokens().len() - 3) as u64; 3]
    );
    assert_eq!(result.maximum_tokens(), result.tokens().len() + 2);
    assert!(result.position(result.maximum_tokens()).is_err());
    assert_eq!(result.source_sha256(), p.source_sha256());
}

#[test]
fn non_square_thw_uses_signed_decode_delta_without_changing_physical_frontier() {
    let plan = PH3ImagePromptExpansionImageTokens {
        tokens: vec![0; 13],
        runs: vec![PH3ImagePromptExpansionImageRun {
            token_start: 5,
            rows: 6,
            height: 2,
            width: 3,
            position_start: 5,
        }],
        maximum_tokens: 16,
        rope_delta: -3,
        source_sha256: Default::default(),
    };
    let expected = [
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
        [5, 5, 6],
        [5, 5, 7],
        [5, 6, 5],
        [5, 6, 6],
        [5, 6, 7],
        [8, 8, 8],
        [9, 9, 9],
        [10, 10, 10],
        [11, 11, 11],
        [12, 12, 12],
    ];
    for (physical, position) in expected.into_iter().enumerate() {
        assert_eq!(plan.position(physical).unwrap(), position);
    }
}

#[test]
fn ordered_images_keep_separate_feature_rows_and_cumulative_text_coordinates() {
    let (p, _) = protocol(&[]);
    let result = p
        .encode_image_input(&render(&p, 2), &[grid(4, 6), grid(2, 4)], 2)
        .unwrap();
    let a = &result.runs[0];
    let b = &result.runs[1];
    assert_eq!(result.image_row(a.token_start + 5), Some((0, 5)));
    assert_eq!(result.image_row(b.token_start), Some((1, 0)));
    assert_eq!(result.image_row(b.token_start + 1), Some((1, 1)));
    let gap = b.token_start - (a.token_start + a.rows);
    assert_eq!(b.position_start, a.position_start + 3 + gap);
    assert_eq!(
        result.position(b.token_start + 1).unwrap(),
        [
            b.position_start as u64,
            b.position_start as u64,
            b.position_start as u64 + 1
        ]
    );
    assert_eq!(result.rope_delta(), -3);
}

#[test]
fn foreign_sources_extra_markers_and_incomplete_merge_cells_never_bind_images() {
    let (p, _) = protocol(&[]);
    let input = render(&p, 1);
    let (foreign, _) = protocol(&["different source"]);
    assert!(foreign
        .encode_image_input(&input, &[grid(4, 6)], 2)
        .is_err());
    assert!(p.encode_image_input(&input, &[], 2).is_err());
    assert!(p.encode_image_input(&input, &[grid(3, 6)], 2).is_err());
    assert!(p.encode_image_input(&input, &[grid(4, 6)], 0).is_err());
    let injection = p
        .render(&PH3ImagePromptExpansionRequest::edit("literal <|image_pad|>", 1).unwrap())
        .unwrap();
    assert!(p.encode_image_input(&injection, &[grid(4, 6)], 2).is_err());
    let text = p
        .render(&PH3ImagePromptExpansionRequest::text_to_image("cat", 64, 64).unwrap())
        .unwrap();
    assert!(p.encode_image_input(&text, &[grid(4, 6)], 2).is_err());
}

#[test]
fn actual_expanded_context_and_rendered_byte_limits_reject_before_model_work() {
    let (p, _) = protocol(&[]);
    let rendered = render(&p, 1);
    let original = p.encode_image_input(&rendered, &[grid(4, 6)], 2).unwrap();
    let (source, _) = source(&[]);
    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    config["model_max_length"] = (original.tokens().len() + 1).into();
    let mut bounded = limits();
    bounded.max_generated_tokens = 2;
    let p = PH3ImagePromptExpansionProtocol::from_source_bytes(
        &source,
        &serde_json::to_vec(&config).unwrap(),
        TEMPLATE,
        bounded,
    )
    .unwrap();
    let rendered = render(&p, 1);
    assert!(p
        .encode_image_input(&rendered, &[grid(4, 6)], 2)
        .unwrap_err()
        .to_string()
        .contains("complete generation"));
    let (p, _) = protocol(&[]);
    assert!(p
        .encode_image_input(&render(&p, 1), &[grid(256, 256)], 2)
        .unwrap_err()
        .to_string()
        .contains("metadata limit"));
}
