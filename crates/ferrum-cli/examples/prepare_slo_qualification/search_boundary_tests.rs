//! Regression for a real tokenizer boundary observed by the existing Rust
//! prompt auditor. This model exercises only search; acceptance still calls eval.
use super::*;

#[test]
fn search_corrects_zero_padding_boundary_using_actual_nonzero_points() {
    let mut measured = Vec::new();
    let hit = bounded_search(1984, 16384, 128, |n| {
        assert!(!measured.iter().any(|&(prior, _)| prior == n));
        let actual = if n == 0 { 98 } else { 99 + n };
        measured.push((n, actual));
        Ok(actual)
    })
    .unwrap();
    assert_eq!(hit.repeats, 1885);
    assert_eq!(hit.evaluations, measured.len());
    assert!(measured.len() <= 128);
    assert!(
        measured.contains(&(1775, 1874)),
        "correction uses the first proposal's actual returned count"
    );
    assert!(
        measured.contains(&(hit.repeats, 1984)),
        "interpolation alone never qualifies an input"
    );
}
