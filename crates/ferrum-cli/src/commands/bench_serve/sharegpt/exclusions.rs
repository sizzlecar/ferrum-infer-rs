//! Bounded, content-audited separation of development and held-out conversations.

use super::{
    sharegpt_first_pair, sharegpt_original_id, visit_sharegpt_records, FerrumError, Result,
    ShareGptReadLimits, ShareGptRecord,
};
use ferrum_bench_core::dataset::{ShareGptExclusionEvidence, ShareGptExclusionSource};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::path::PathBuf;

pub(super) const MAX_SOURCES: usize = 256;
const MAX_IDENTITIES: usize = 1_000_000;
const MAX_ID_BYTES: usize = 64 * 1024 * 1024;
const MATCHING: &str = "original_id_or_unicode_whitespace_collapsed_prompt_sha256_v1";

#[derive(Default)]
pub(super) struct Exclusions {
    ids: HashSet<String>,
    prompts: HashSet<[u8; 32]>,
    sources: Vec<ShareGptExclusionSource>,
    id_bytes: usize,
}

impl Exclusions {
    pub(super) fn load(paths: &[PathBuf]) -> Result<Self> {
        if paths.len() > MAX_SOURCES {
            return Err(FerrumError::model("too many ShareGPT exclusion sources"));
        }
        let mut result = Self::default();
        for path in paths {
            let file = File::open(path).map_err(|error| {
                FerrumError::model(format!(
                    "open ShareGPT exclusion {}: {error}",
                    path.display()
                ))
            })?;
            let receipt =
                visit_sharegpt_records(file, ShareGptReadLimits::default(), |_, record| {
                    let ShareGptRecord::Value(value) = record else {
                        return Err(FerrumError::model("malformed ShareGPT exclusion record"));
                    };
                    let id = sharegpt_original_id(&value);
                    let pair = sharegpt_first_pair(&value);
                    // Silently skipping an unusable history record could admit a
                    // previously observed request into the held-out pool.
                    if id.is_none() && pair.is_none() {
                        return Err(FerrumError::model(
                            "ShareGPT exclusion record has neither an ID nor a first pair",
                        ));
                    }
                    if let Some(id) = id {
                        if !result.ids.contains(&id) {
                            if result.ids.len() == MAX_IDENTITIES
                                || id.len() > MAX_ID_BYTES.saturating_sub(result.id_bytes)
                            {
                                return Err(FerrumError::model(
                                    "ShareGPT exclusion ID budget exceeded",
                                ));
                            }
                            result.id_bytes += id.len();
                            result.ids.insert(id);
                        }
                    }
                    if let Some(pair) = pair {
                        let fingerprint = normalized_prompt_fingerprint(pair.prompt);
                        if !result.prompts.contains(&fingerprint) {
                            if result.prompts.len() == MAX_IDENTITIES {
                                return Err(FerrumError::model(
                                    "ShareGPT exclusion prompt budget exceeded",
                                ));
                            }
                            result.prompts.insert(fingerprint);
                        }
                    }
                    Ok(())
                })?;
            result.sources.push(ShareGptExclusionSource {
                source_path: path.display().to_string(),
                source_sha256: receipt.source_sha256,
                records: receipt.records,
            });
        }
        Ok(result)
    }

    pub(super) fn evidence(&self) -> Option<ShareGptExclusionEvidence> {
        (!self.sources.is_empty()).then(|| ShareGptExclusionEvidence {
            matching: MATCHING.into(),
            sources: self.sources.clone(),
            unique_original_ids: self.ids.len(),
            unique_normalized_prompts: self.prompts.len(),
        })
    }

    pub(super) fn contains_id(&self, id: Option<&str>) -> bool {
        id.is_some_and(|id| self.ids.contains(id))
    }

    pub(super) fn contains_prompt(&self, prompt: &str) -> bool {
        !self.prompts.is_empty()
            && self
                .prompts
                .contains(&normalized_prompt_fingerprint(prompt))
    }
}

/// Trim and collapse Unicode whitespace only. Case and non-whitespace bytes
/// remain significant; this fingerprint is a separation rule, not a text edit.
fn normalized_prompt_fingerprint(prompt: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    for (index, word) in prompt.split_whitespace().enumerate() {
        if index > 0 {
            hash.update(b" ");
        }
        hash.update(word.as_bytes());
    }
    hash.finalize().into()
}
