//! Explicit, content-bound product documentation reviews from the release catalog.
//! These records describe reviewed reach, never successful execution or performance.

use ferrum_bench_core::release_regression::{ChangeArea, Impact};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReadmeReview {
    pub path: String,
    pub before_sha256: String,
    pub after_sha256: String,
    pub areas: Vec<ChangeArea>,
    pub rationale: String,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub(super) struct ReviewMismatch {
    path: String,
    expected_before_sha256: String,
    expected_after_sha256: String,
    actual_before_sha256: String,
    actual_after_sha256: String,
}

#[derive(Debug, Default)]
pub(super) struct ReviewOutcome {
    pub applied: Vec<ReadmeReview>,
    pub unmatched: Vec<ReviewMismatch>,
}

fn readme(path: &str) -> bool {
    matches!(path, "README.md" | "README_zh.md")
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn validate(reviews: &[ReadmeReview]) -> Result<(), String> {
    let mut paths = BTreeSet::new();
    for review in reviews {
        if !readme(&review.path) || !paths.insert(&review.path) {
            return Err(format!(
                "unknown or duplicate README review path {:?}",
                review.path
            ));
        }
        for hash in [&review.before_sha256, &review.after_sha256] {
            if hash.len() != 64
                || !hash
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(format!("invalid README review SHA256 for {}", review.path));
            }
        }
        if review.rationale.trim().is_empty() {
            return Err(format!(
                "README review requires a rationale for {}",
                review.path
            ));
        }
        let areas: BTreeSet<_> = review.areas.iter().copied().collect();
        if areas.is_empty()
            || areas.len() != review.areas.len()
            || areas.iter().any(|area| {
                !matches!(
                    area,
                    ChangeArea::Build
                        | ChangeArea::Template
                        | ChangeArea::Termination
                        | ChangeArea::Structured
                        | ChangeArea::Tools
                )
            })
        {
            return Err(format!(
                "README review requires distribution/protocol areas for {}",
                review.path
            ));
        }
    }
    Ok(())
}

pub(super) fn take_reviews(catalog: &mut Value) -> Result<Vec<ReadmeReview>, String> {
    let object = catalog.as_object_mut().ok_or("catalog must be an object")?;
    let Some(value) = object.remove("readme_reviews") else {
        return Ok(Vec::new());
    };
    let reviews: Vec<ReadmeReview> = serde_json::from_value(value)
        .map_err(|error| format!("invalid README reviews: {error}"))?;
    validate(&reviews)?;
    Ok(reviews)
}

/// The reader must return both immutable Git blobs for the actual plan revisions.
/// Validate every applicable review before mutating any impact. Old entries for
/// unchanged README files are inert; they are not read against a different base.
pub(super) fn apply_reviews(
    impact: &mut Impact,
    reviews: &[ReadmeReview],
    mut read: impl FnMut(&str) -> Result<(Vec<u8>, Vec<u8>), String>,
) -> Result<ReviewOutcome, String> {
    validate(reviews)?;
    let mut outcome = ReviewOutcome::default();
    for review in reviews {
        if !impact.paths.iter().any(|entry| entry.path == review.path) {
            continue;
        }
        let (before, after) = read(&review.path)?;
        let actual_before_sha256 = digest(&before);
        let actual_after_sha256 = digest(&after);
        if actual_before_sha256 != review.before_sha256
            || actual_after_sha256 != review.after_sha256
        {
            outcome.unmatched.push(ReviewMismatch {
                path: review.path.clone(),
                expected_before_sha256: review.before_sha256.clone(),
                expected_after_sha256: review.after_sha256.clone(),
                actual_before_sha256,
                actual_after_sha256,
            });
            continue;
        }
        outcome.applied.push(review.clone());
    }
    for review in &outcome.applied {
        let entry = impact
            .paths
            .iter_mut()
            .find(|entry| entry.path == review.path)
            .expect("applicable review has a changed path");
        entry.areas = review.areas.clone();
        entry.reason = format!(
            "content-bound README product review: {}; retain declared distribution/protocol checks; no execution is certified",
            review.rationale
        );
    }
    // A PR can use a different baseline than formal release planning. Preserve
    // the unreviewed contract and report its actual hashes instead of suppressing
    // the entire planning report. Release gating still rejects this gap.
    if !outcome.unmatched.is_empty() {
        impact.product_contract_changed = true;
    }
    // Never let a review for one translation excuse a different unreviewed file.
    // Existing automatic content proofs can already have cleared this flag.
    if impact.product_contract_changed
        && impact
            .paths
            .iter()
            .filter(|entry| readme(&entry.path))
            .all(|entry| {
                outcome
                    .applied
                    .iter()
                    .any(|review| review.path == entry.path)
            })
        && !outcome.applied.is_empty()
    {
        impact.product_contract_changed = false;
    }
    if !outcome.applied.is_empty() {
        impact.areas = ChangeArea::ALL
            .into_iter()
            .filter(|area| {
                !impact.unknown_paths.is_empty()
                    || impact.paths.iter().any(|entry| entry.areas.contains(area))
            })
            .collect();
    }
    Ok(outcome)
}

#[cfg(test)]
#[path = "readme_tests.rs"]
mod tests;
