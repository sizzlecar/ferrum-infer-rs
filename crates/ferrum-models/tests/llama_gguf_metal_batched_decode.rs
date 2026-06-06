#![cfg(all(target_os = "macos", feature = "metal"))]

use std::path::PathBuf;

use ferrum_models::gguf_engine_loader::load_gguf_decoder;
use ferrum_types::Device;

fn llama_gguf_path() -> PathBuf {
    std::env::var_os("FERRUM_TEST_LLAMA_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/Users/chejinxuan/ferrum-bench/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            )
        })
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap()
}

fn dot_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    (dot / (aa.sqrt() * bb.sqrt()).max(1e-30)) as f32
}

fn top_k(xs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = xs.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs
}

fn assert_close_logits(label: &str, sequential: &[f32], batched: &[f32], vocab: usize) -> u32 {
    assert_eq!(sequential.len(), vocab, "{label}: sequential logits len");
    assert_eq!(batched.len(), vocab, "{label}: batched logits len");
    assert!(
        sequential.iter().all(|v| v.is_finite()) && batched.iter().all(|v| v.is_finite()),
        "{label}: non-finite logits"
    );
    let seq_top = argmax(sequential);
    let bat_top = argmax(batched);
    let cosine = dot_cosine(sequential, batched);
    eprintln!(
        "{label}: seq_top={seq_top} bat_top={bat_top} cosine={cosine:.6}\n  seq_top5={:?}\n  bat_top5={:?}",
        top_k(sequential, 5),
        top_k(batched, 5)
    );
    assert_eq!(seq_top, bat_top, "{label}: batched top token diverged");
    assert!(
        cosine > 0.999,
        "{label}: batched logits diverged, cosine={cosine:.6}"
    );
    seq_top as u32
}

#[test]
#[ignore = "loads local Llama-3.1-8B GGUF and compares Metal sequential vs batched decode"]
fn llama31_8b_metal_batched_decode_matches_sequential_decode() {
    let path = llama_gguf_path();
    assert!(path.is_file(), "missing test model: {}", path.display());

    let mut model = load_gguf_decoder(&path, &Device::Metal).expect("load Llama GGUF on Metal");
    let vocab = model.config().vocab_size;

    // Valid Llama-3.x token ids. Exact text is unimportant here; the probe
    // compares two execution paths for identical token/KV states.
    let prompt_a: Vec<u32> = vec![128000, 9906, 11, 1917, 13];
    let prompt_b: Vec<u32> = vec![128000, 791, 4062, 374, 220, 16, 13];
    assert!(prompt_a
        .iter()
        .chain(&prompt_b)
        .all(|&t| (t as usize) < vocab));

    let seq_a_prefill = model.prefill("seq_a", &prompt_a);
    let seq_b_prefill = model.prefill("seq_b", &prompt_b);
    let bat_a_prefill = model.prefill("bat_a", &prompt_a);
    let bat_b_prefill = model.prefill("bat_b", &prompt_b);

    let mut next_a = assert_close_logits("prefill_a", &seq_a_prefill, &bat_a_prefill, vocab);
    let mut next_b = assert_close_logits("prefill_b", &seq_b_prefill, &bat_b_prefill, vocab);
    let mut pos_a = prompt_a.len() as u32;
    let mut pos_b = prompt_b.len() as u32;

    for step in 1..=3 {
        let seq_a = model.decode("seq_a", next_a, pos_a);
        let seq_b = model.decode("seq_b", next_b, pos_b);
        let batch = vec![
            ("bat_a".to_string(), next_a, pos_a),
            ("bat_b".to_string(), next_b, pos_b),
        ];
        let batched = model.decode_batch(&batch);
        assert_eq!(batched.len(), 2, "decode_batch row count");

        next_a = assert_close_logits(&format!("decode_step_{step}_a"), &seq_a, &batched[0], vocab);
        next_b = assert_close_logits(&format!("decode_step_{step}_b"), &seq_b, &batched[1], vocab);
        pos_a += 1;
        pos_b += 1;
    }
}
