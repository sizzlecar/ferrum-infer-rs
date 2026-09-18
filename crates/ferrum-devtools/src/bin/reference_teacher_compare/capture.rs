use anyhow::{ensure, Context, Result};
use ferrum_interfaces::vnext::TokenSpanWork;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

pub(super) struct Wave {
    pub token: u32,
    pub logits: Vec<f32>,
    pub manifest_sha256: String,
    pub raw_sha256: String,
}
pub(super) struct Capture {
    pub prompt: Vec<u32>,
    pub waves: Vec<Wave>,
    pub plan_sha256: String,
    pub teacher_sha256: String,
}

pub(super) fn token_digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn count(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .context("invalid capture count")
}

impl Capture {
    pub fn read(directory: &Path) -> Result<Self> {
        let plan_bytes = fs::read(directory.join("plan.json"))?;
        let plan: Value = serde_json::from_slice(&plan_bytes)?;
        let teacher = &plan["teacher_forcing"];
        ensure!(
            plan["schema_version"] == 4
                && teacher["mode"] == "canonical-history"
                && teacher["encoding"] == "u32-le"
                && teacher["prompt_file"] == "teacher-prompt.json",
            "capture is not a complete typed teacher history"
        );
        let wave_count = count(&teacher["token_count"])?;
        ensure!(
            wave_count > 0
                && plan["maximum_prefill_waves"] == 1
                && plan["maximum_decode_waves"] == wave_count - 1
                && plan["capture_product_output"] == true,
            "teacher capture counts differ from the complete history"
        );
        let prompt: Value =
            serde_json::from_slice(&fs::read(directory.join("teacher-prompt.json"))?)?;
        let tokens: Vec<u32> = serde_json::from_value(prompt["token_ids"].clone())?;
        ensure!(
            !tokens.is_empty()
                && prompt["schema_version"] == 1
                && prompt["encoding"] == "u32-le"
                && prompt["token_count"] == tokens.len()
                && prompt["token_ids_sha256"] == token_digest(&tokens),
            "invalid canonical prompt token evidence"
        );
        let request = prompt["request_id"]
            .as_str()
            .filter(|s| !s.is_empty())
            .context("missing prompt request identity")?;
        let expected_names: BTreeSet<_> = (0..wave_count)
            .map(|index| {
                if index == 0 {
                    "wave-0000.json".to_owned()
                } else {
                    format!("decode-wave-{:04}.json", index - 1)
                }
            })
            .collect();
        let mut observed_names = BTreeSet::new();
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json")
                    && (name.starts_with("wave-") || name.starts_with("decode-wave-"))
                {
                    ensure!(
                        entry.file_type()?.is_file(),
                        "wave manifest must be a regular file"
                    );
                    observed_names.insert(name.to_owned());
                }
            }
        }
        ensure!(
            observed_names == expected_names,
            "capture has missing or extra teacher waves"
        );
        let mut history = tokens.clone();
        let mut teacher_tokens = Vec::new();
        let mut waves = Vec::new();
        let mut vocabulary = None;
        for index in 0..wave_count {
            let (name, kind, capture_index) = if index == 0 {
                ("wave-0000.json".to_owned(), "prefill", 0)
            } else {
                (
                    format!("decode-wave-{:04}.json", index - 1),
                    "decode",
                    index - 1,
                )
            };
            let bytes = fs::read(directory.join(name))?;
            let wave: Value = serde_json::from_slice(&bytes)?;
            ensure!(
                wave["schema_version"] == 4
                    && wave["participant_count"] == 1
                    && wave["capture_index"] == capture_index
                    && wave["wave_kind"] == kind
                    && wave["teacher_forced_decision"]["token_index"] == index,
                "wave does not match its canonical teacher decision"
            );
            for field in [
                "plan_id",
                "plan_hash",
                "model_id",
                "family_fingerprint",
                "program_fingerprint",
                "run_id",
            ] {
                ensure!(
                    plan[field].as_str().is_some_and(|value| !value.is_empty())
                        && wave[field] == plan[field],
                    "wave {field} changed during the teacher capture"
                );
            }
            let token = wave["teacher_forced_decision"]["token_id"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .context("invalid teacher token")?;
            let outputs = wave["product_outputs"]
                .as_array()
                .context("missing captured output")?;
            ensure!(
                outputs.len() == 1,
                "expected exactly one complete product distribution"
            );
            let output = &outputs[0];
            ensure!(
                output["output_mode"] == "full-logits"
                    && output["participant_index"] == 0
                    && output["request_id"] == request,
                "product output owner or representation differs"
            );
            let span = &output["token_span"];
            let start = count(&span["immediate_start_token"])?;
            let end = count(&span["immediate_end_token"])?;
            let expected = TokenSpanWork::from_token_ids_with_fit(
                &history,
                start..end,
                count(&span["fit_input_tokens"])?,
            )?;
            ensure!(
                serde_json::to_value(expected)? == *span && end == history.len(),
                "captured distribution does not follow the exact canonical history"
            );
            ensure!(
                output["output_layout"]["element_type"] == "f32",
                "reference comparison requires F32 product logits"
            );
            let elements = count(&output["output_layout"]["element_count"])?;
            ensure!(
                elements > 0 && (token as usize) < elements,
                "teacher target is outside captured vocabulary"
            );
            if let Some(vocabulary) = vocabulary {
                ensure!(elements == vocabulary, "captured vocabulary changed");
            }
            vocabulary = Some(elements);
            ensure!(
                history.iter().all(|token| (*token as usize) < elements),
                "prompt is outside captured vocabulary"
            );
            let name = output["raw_file"].as_str().context("missing logits file")?;
            ensure!(
                !name.contains(['/', '\\'])
                    && Path::new(name).file_name().and_then(|name| name.to_str()) == Some(name),
                "logits must be stored in an adjacent file"
            );
            let raw = fs::read(directory.join(name))?;
            let raw_sha256 = super::sha256(&raw);
            ensure!(
                elements.checked_mul(4) == Some(raw.len())
                    && output["raw_bytes"] == raw.len()
                    && output["raw_sha256"] == raw_sha256,
                "logits bytes differ from their capture manifest"
            );
            let logits: Vec<_> = raw
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            ensure!(
                logits.iter().all(|value| value.is_finite()),
                "captured logits contain non-finite values"
            );
            history.push(token);
            teacher_tokens.push(token);
            waves.push(Wave {
                token,
                logits,
                manifest_sha256: super::sha256(&bytes),
                raw_sha256,
            });
        }
        let teacher_sha256 = token_digest(&teacher_tokens);
        ensure!(
            teacher["token_ids_sha256"] == teacher_sha256,
            "complete teacher decisions differ from the planned history"
        );
        Ok(Self {
            prompt: tokens,
            waves,
            plan_sha256: super::sha256(&plan_bytes),
            teacher_sha256,
        })
    }
}
