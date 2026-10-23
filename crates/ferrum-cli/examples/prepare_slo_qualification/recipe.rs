use anyhow::{bail, ensure, Context, Result};
use ferrum_types::{ModelOutputProtocol, ReasoningEffort};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    path::{Component, PathBuf},
};

pub(super) const MAX_RECIPE_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Recipe {
    pub schema_version: u32,
    pub model_snapshot: PathBuf,
    pub template_source: TemplateSource,
    pub output_protocol: ModelOutputProtocol,
    pub context_tokens: usize,
    pub kv_block_tokens: usize,
    pub seed: u64,
    pub instruction: String,
    pub padding_unit: String,
    pub client: ClientSettings,
    #[serde(default)]
    pub limits: Limits,
    pub stages: Vec<Stage>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum TemplateSource {
    TokenizerConfig,
    ChatTemplateJinja,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ClientSettings {
    pub base_url: String,
    pub model: String,
    pub deadline_ms: u64,
    pub request_policy: RequestPolicy,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestPolicy {
    pub sampling: Sampling,
    pub ignore_eos: bool,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<ReasoningEffort>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sampling {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Stage {
    pub name: String,
    pub output_dir: PathBuf,
    pub groups: Vec<Group>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Group {
    pub count: usize,
    pub target_rendered_tokens: usize,
    pub max_tokens: u32,
    pub first_offset_ms: u64,
    pub interval_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Limits {
    pub max_requests: usize,
    pub max_source_bytes: usize,
    pub max_raw_prompt_bytes: usize,
    pub max_rendered_bytes: usize,
    pub max_output_bytes: usize,
    pub max_candidate_bytes: usize,
    pub max_repeats: usize,
    pub max_evaluations_per_request: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_requests: 1024,
            max_source_bytes: 64 * 1024 * 1024,
            max_raw_prompt_bytes: 64 * 1024,
            max_rendered_bytes: 128 * 1024,
            max_output_bytes: 64 * 1024 * 1024,
            max_candidate_bytes: 256 * 1024 * 1024,
            max_repeats: 16384,
            max_evaluations_per_request: 128,
        }
    }
}

impl Recipe {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema_version == 1, "unsupported recipe schema");
        ensure!(
            !self.model_snapshot.as_os_str().is_empty(),
            "model snapshot is required"
        );
        ensure!(
            (1..=131072).contains(&self.context_tokens) && self.kv_block_tokens > 0,
            "invalid declared context/block geometry"
        );
        ensure!(
            !self.instruction.is_empty() && self.instruction.len() <= 4096,
            "instruction must have 1..4096 bytes"
        );
        ensure!(
            !self.padding_unit.is_empty()
                && self.padding_unit.is_ascii()
                && self.padding_unit.len() <= 256,
            "padding_unit must have 1..256 ASCII bytes"
        );
        let l = &self.limits;
        ensure!(
            (1..=1024).contains(&l.max_requests)
                && (1..=64 * 1024 * 1024).contains(&l.max_source_bytes)
                && (1..=1024 * 1024).contains(&l.max_raw_prompt_bytes)
                && (1..=2 * 1024 * 1024).contains(&l.max_rendered_bytes)
                && (1..=128 * 1024 * 1024).contains(&l.max_output_bytes)
                && (1..=1024 * 1024 * 1024).contains(&l.max_candidate_bytes)
                && l.max_repeats <= 131072
                && (1..=512).contains(&l.max_evaluations_per_request),
            "recipe exceeds hard resource bounds"
        );
        ensure!(
            !self.client.base_url.is_empty()
                && self.client.base_url.len() <= 4096
                && !self.client.model.is_empty()
                && self.client.model.len() <= 4096
                && self.client.deadline_ms > 0,
            "invalid client binding"
        );
        let s = &self.client.request_policy.sampling;
        ensure!(
            s.temperature.is_finite()
                && s.temperature >= 0.0
                && s.top_p.is_none_or(|p| p.is_finite() && p > 0.0 && p <= 1.0)
                && s.repetition_penalty
                    .is_none_or(|p| p.is_finite() && p > 0.0),
            "invalid sampling policy"
        );
        ensure!(
            !self.stages.is_empty() && self.stages.len() <= 32,
            "expected 1..32 finite stages"
        );
        let mut names = BTreeSet::new();
        let mut paths: BTreeSet<&PathBuf> = BTreeSet::new();
        let mut count = 0usize;
        for stage in &self.stages {
            ensure!(
                !stage.name.is_empty()
                    && stage.name.len() <= 64
                    && stage
                        .name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
                    && names.insert(&stage.name),
                "invalid or duplicate stage name"
            );
            // Runtime paths may target another host; lexical absolute normalized
            // paths reject common aliasing without touching that host's FS.
            ensure!(
                stage.output_dir.is_absolute()
                    && !stage
                        .output_dir
                        .components()
                        .any(|c| matches!(c, Component::ParentDir | Component::CurDir))
                    && paths
                        .iter()
                        .all(|prior| !prior.starts_with(&stage.output_dir)
                            && !stage.output_dir.starts_with(prior))
                    && paths.insert(&stage.output_dir),
                "stage output paths must be unique absolute normalized paths"
            );
            ensure!(
                !stage.groups.is_empty() && stage.groups.len() <= 1024,
                "empty or oversized groups"
            );
            let mut last_offset = None;
            for group in &stage.groups {
                ensure!(
                    group.count > 0 && group.max_tokens > 0 && group.target_rendered_tokens > 0,
                    "request count, target and cap must be positive"
                );
                count = count
                    .checked_add(group.count)
                    .context("request count overflow")?;
                ensure!(count <= l.max_requests, "request budget exceeded");
                let context = group
                    .target_rendered_tokens
                    .checked_add(group.max_tokens as usize)
                    .context("context overflow")?;
                ensure!(
                    context <= self.context_tokens,
                    "rendered prompt plus cap exceeds declared context"
                );
                let offset = u64::try_from(group.count - 1)
                    .ok()
                    .and_then(|n| n.checked_mul(group.interval_ms))
                    .and_then(|n| n.checked_add(group.first_offset_ms))
                    .context("arrival offset overflow")?;
                let end_ms = offset
                    .checked_add(self.client.deadline_ms)
                    .context("arrival plus deadline overflow")?;
                ensure!(
                    std::time::Instant::now()
                        .checked_add(std::time::Duration::from_millis(end_ms))
                        .is_some(),
                    "arrival deadline exceeds host monotonic clock range"
                );
                if last_offset.is_some_and(|last| group.first_offset_ms < last) {
                    bail!("arrival offsets must be nondecreasing within each stage");
                }
                last_offset = Some(offset);
            }
        }
        Ok(())
    }
}
