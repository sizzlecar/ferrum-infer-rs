use super::{hash, recipe::*};
use anyhow::{bail, ensure, Context, Result};
use ferrum_server::{
    chat_template::{
        render_chat_prompt_with_model_template_options, ChatTemplateOptions, ModelChatTemplate,
    },
    openai::ChatMessage,
};
use ferrum_tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub(super) struct SearchHit {
    pub repeats: usize,
    pub evaluations: usize,
}

/// A finite heuristic search, not a monotonicity or reachability proof. Only an
/// actual equality returned by eval qualifies a candidate.
pub(super) fn bounded_search(
    target: usize,
    max_repeats: usize,
    max_evaluations: usize,
    mut eval: impl FnMut(usize) -> Result<usize>,
) -> Result<SearchHit> {
    ensure!(
        max_evaluations > 0 && max_evaluations <= 512 && max_repeats <= 131072,
        "invalid finite search budget"
    );
    let mut visited = Vec::with_capacity(max_evaluations);
    fn observe(
        n: usize,
        max_evaluations: usize,
        visited: &mut Vec<usize>,
        eval: &mut impl FnMut(usize) -> Result<usize>,
    ) -> Result<Option<usize>> {
        if visited.contains(&n) || visited.len() == max_evaluations {
            return Ok(None);
        }
        visited.push(n);
        Ok(Some(eval(n)?))
    }
    let base =
        observe(0, max_evaluations, &mut visited, &mut eval)?.context("empty evaluation budget")?;
    if base == target {
        return Ok(SearchHit {
            repeats: 0,
            evaluations: 1,
        });
    }
    let probe = max_repeats.min(16);
    let probed = if probe > 0 {
        observe(probe, max_evaluations, &mut visited, &mut eval)?
    } else {
        None
    };
    if probed == Some(target) {
        return Ok(SearchHit {
            repeats: probe,
            evaluations: visited.len(),
        });
    }
    let estimate = probed
        .filter(|&n| n > base)
        .map_or(0, |n| {
            target.saturating_sub(base).saturating_mul(probe) / (n - base)
        })
        .min(max_repeats);
    // Enumeration is bounded even when all legal positions were already tried.
    for distance in 0..=max_repeats {
        for candidate in [
            estimate.checked_add(distance).filter(|&n| n <= max_repeats),
            (distance != 0)
                .then(|| estimate.checked_sub(distance))
                .flatten(),
        ]
        .into_iter()
        .flatten()
        {
            if let Some(actual) = observe(candidate, max_evaluations, &mut visited, &mut eval)? {
                if actual == target {
                    return Ok(SearchHit {
                        repeats: candidate,
                        evaluations: visited.len(),
                    });
                }
            }
            if visited.len() >= max_evaluations {
                break;
            }
        }
        if visited.len() >= max_evaluations {
            break;
        }
    }
    bail!("search exhausted after {} evaluations; exact target {target} was not reached under this recipe", visited.len())
}

pub(super) fn nonce(seed: u64, stage: &str, index: usize) -> String {
    let mut digest = Sha256::new();
    digest.update(b"ferrum.synthetic.qualification.v1\0");
    digest.update(seed.to_le_bytes());
    digest.update((stage.len() as u64).to_le_bytes());
    digest.update(stage.as_bytes());
    digest.update((index as u64).to_le_bytes());
    format!("{:x}", digest.finalize())
}

pub(super) struct GenerationBudget {
    remaining_bytes: usize,
    pub evaluations: usize,
    pub bytes_charged: usize,
}
impl GenerationBudget {
    pub fn new(bytes: usize) -> Self {
        Self {
            remaining_bytes: bytes,
            evaluations: 0,
            bytes_charged: 0,
        }
    }
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.remaining_bytes = self
            .remaining_bytes
            .checked_sub(bytes)
            .context("total candidate byte budget exhausted")?;
        self.bytes_charged = self
            .bytes_charged
            .checked_add(bytes)
            .context("candidate byte accounting overflow")?;
        Ok(())
    }
}

pub(super) struct PreparedPrompt {
    pub prompt: String,
    pub rendered_prompt: String,
    pub token_ids: Vec<u32>,
    pub input_tokens_raw: usize,
    pub repeats: usize,
    pub evaluations: usize,
}

pub(super) fn prepare_prompt(
    recipe: &Recipe,
    stage: &str,
    index: usize,
    target: usize,
    tokenizer: &dyn Tokenizer,
    template: &ModelChatTemplate,
    budget: &mut GenerationBudget,
) -> Result<PreparedPrompt> {
    ensure!(
        target > 0 && target <= recipe.context_tokens,
        "invalid target token length"
    );
    let id = nonce(recipe.seed, stage, index);
    // The varying digest starts the user content; the seed/stage literal is not
    // a large artificial common prefix. Templates may still share prefixes.
    let prefix = format!("{id}\n{}\n", recipe.instruction);
    let available = recipe
        .limits
        .max_raw_prompt_bytes
        .checked_sub(prefix.len())
        .context("nonce/instruction exceeds raw prompt limit")?;
    ensure!(!recipe.padding_unit.is_empty(), "empty padding unit");
    let max_repeats = recipe
        .limits
        .max_repeats
        .min(available / recipe.padding_unit.len());
    let options = ChatTemplateOptions {
        enable_thinking: recipe.client.request_policy.enable_thinking,
        reasoning_effort: recipe.client.request_policy.reasoning_effort,
        now_override: None,
    };
    let mut accepted = None;
    let hit = bounded_search(
        target,
        max_repeats,
        recipe.limits.max_evaluations_per_request,
        |repeats| {
            let size = recipe
                .padding_unit
                .len()
                .checked_mul(repeats)
                .and_then(|n| n.checked_add(prefix.len()))
                .context("prompt byte overflow")?;
            ensure!(
                size <= recipe.limits.max_raw_prompt_bytes,
                "raw prompt byte limit"
            );
            budget.charge(size)?;
            budget.evaluations = budget
                .evaluations
                .checked_add(1)
                .context("evaluation count overflow")?;
            let mut prompt = String::with_capacity(size);
            prompt.push_str(&prefix);
            for _ in 0..repeats {
                prompt.push_str(&recipe.padding_unit);
            }
            let message: ChatMessage =
                serde_json::from_value(json!({"role":"user", "content":&prompt}))?;
            let rendered_prompt = render_chat_prompt_with_model_template_options(
                &[message],
                &recipe.client.model,
                Some(template),
                &options,
            )?;
            ensure!(
                rendered_prompt.len() <= recipe.limits.max_rendered_bytes,
                "rendered prompt byte limit"
            );
            budget.charge(rendered_prompt.len())?;
            let encoded = tokenizer.encode(&rendered_prompt, true)?;
            budget.charge(
                encoded
                    .len()
                    .checked_mul(std::mem::size_of::<u32>())
                    .context("token bytes overflow")?,
            )?;
            let count = encoded.len();
            if count == target {
                let raw = tokenizer.encode(&prompt, false)?;
                budget.charge(
                    raw.len()
                        .checked_mul(std::mem::size_of::<u32>())
                        .context("raw token bytes overflow")?,
                )?;
                accepted = Some(PreparedPrompt {
                    prompt,
                    rendered_prompt,
                    token_ids: encoded.into_iter().map(u32::from).collect(),
                    input_tokens_raw: raw.len(),
                    repeats,
                    evaluations: 0,
                });
            }
            Ok(count)
        },
    )?;
    let mut prepared = accepted.context("exact search hit lacks actual rendered input")?;
    ensure!(
        prepared.repeats == hit.repeats && prepared.token_ids.len() == target,
        "search result mismatch"
    );
    prepared.evaluations = hit.evaluations;
    Ok(prepared)
}

#[derive(Serialize)]
pub(super) struct WorkRequest {
    pub scenario: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub offset_ms: u64,
    pub source: Value,
}
#[derive(Serialize)]
pub(super) struct ClientConfig {
    pub base_url: String,
    pub model: String,
    pub output_dir: std::path::PathBuf,
    pub requests: Vec<WorkRequest>,
    pub deadline_ms: u64,
    pub temperature: f32,
    pub protocol: &'static str,
    pub request_policy: RequestPolicy,
    pub provenance: Value,
}
#[derive(Serialize)]
pub(super) struct PromptAudit {
    pub source_id: String,
    pub input_tokens_raw: usize,
    pub input_tokens_rendered: usize,
    pub raw_prompt_sha256: String,
    pub rendered_prompt_sha256: String,
    pub rendered_prompt: String,
    pub token_ids: Vec<u32>,
    pub max_tokens: u32,
    pub offset_ms: u64,
    pub context_fits: bool,
    pub padding_repeats: usize,
    pub search_evaluations: usize,
    pub maximum_prior_shared_prefix_tokens: usize,
    pub maximum_prior_potential_reusable_blocks: usize,
}
#[derive(Serialize)]
pub(super) struct StageAudit {
    pub schema_version: u32,
    pub kind: &'static str,
    pub model: String,
    pub stage: String,
    pub context_tokens: usize,
    pub kv_block_tokens: usize,
    pub requests: Vec<PromptAudit>,
    pub ordered_input_sha256: String,
    pub arrival_offsets_sha256: String,
}
pub(super) struct PreparedStage {
    pub client: ClientConfig,
    pub audit: StageAudit,
}

pub(super) fn prepare_stage(
    recipe: &Recipe,
    stage: &Stage,
    tokenizer: &dyn Tokenizer,
    template: &ModelChatTemplate,
    recipe_hash: &str,
    budget: &mut GenerationBudget,
    retained_payload_bytes: &mut usize,
    previous_tokens: &mut Vec<Vec<u32>>,
) -> Result<PreparedStage> {
    recipe.validate()?;
    let count = stage
        .groups
        .iter()
        .try_fold(0usize, |a, g| a.checked_add(g.count))
        .context("stage count overflow")?;
    ensure!(count <= recipe.limits.max_requests, "request budget");
    let mut requests = Vec::with_capacity(count);
    let mut audits = Vec::with_capacity(count);
    for group in &stage.groups {
        for local in 0..group.count {
            let index = requests.len();
            let offset_ms = u64::try_from(local)
                .ok()
                .and_then(|n| n.checked_mul(group.interval_ms))
                .and_then(|n| n.checked_add(group.first_offset_ms))
                .context("arrival offset overflow")?;
            let prepared = prepare_prompt(
                recipe,
                &stage.name,
                index,
                group.target_rendered_tokens,
                tokenizer,
                template,
                budget,
            )?;
            let context_fits = prepared
                .token_ids
                .len()
                .checked_add(group.max_tokens as usize)
                .is_some_and(|n| n <= recipe.context_tokens);
            ensure!(
                context_fits,
                "actual rendered prompt plus cap exceeds context"
            );
            let bytes = prepared
                .prompt
                .len()
                .checked_add(prepared.rendered_prompt.len())
                .and_then(|n| {
                    prepared
                        .token_ids
                        .len()
                        .checked_mul(8)
                        .and_then(|ids| n.checked_add(ids))
                })
                .context("retained prompt payload overflow")?;
            *retained_payload_bytes = retained_payload_bytes
                .checked_add(bytes)
                .context("retained payload accounting overflow")?;
            ensure!(
                *retained_payload_bytes <= recipe.limits.max_output_bytes,
                "retained prompt payload budget exceeded"
            );
            let mut shared = 0;
            let mut blocks = 0;
            for prior in previous_tokens.iter() {
                let common = prior
                    .iter()
                    .zip(&prepared.token_ids)
                    .take_while(|(a, b)| a == b)
                    .count();
                shared = shared.max(common);
                blocks = blocks.max(
                    common.min(prior.len().min(prepared.token_ids.len()).saturating_sub(1))
                        / recipe.kv_block_tokens,
                );
            }
            previous_tokens.push(prepared.token_ids.clone());
            let source_id = format!(
                "synthetic:{}:{}",
                stage.name,
                nonce(recipe.seed, &stage.name, index)
            );
            let raw_hash = hash(prepared.prompt.as_bytes());
            let source = json!({"kind":"synthetic_public_geometry", "source_id":source_id,
                "recipe_sha256":recipe_hash,"seed":recipe.seed,"stage":stage.name,"index":index,
                "prompt_sha256":raw_hash,"input_tokens_raw":prepared.input_tokens_raw,
                "target_rendered_tokens":group.target_rendered_tokens});
            audits.push(PromptAudit {
                source_id,
                input_tokens_raw: prepared.input_tokens_raw,
                input_tokens_rendered: prepared.token_ids.len(),
                raw_prompt_sha256: raw_hash,
                rendered_prompt_sha256: hash(prepared.rendered_prompt.as_bytes()),
                rendered_prompt: prepared.rendered_prompt,
                token_ids: prepared.token_ids,
                max_tokens: group.max_tokens,
                offset_ms,
                context_fits,
                padding_repeats: prepared.repeats,
                search_evaluations: prepared.evaluations,
                maximum_prior_shared_prefix_tokens: shared,
                maximum_prior_potential_reusable_blocks: blocks,
            });
            requests.push(WorkRequest {
                scenario: stage.name.clone(),
                prompt: prepared.prompt,
                max_tokens: group.max_tokens,
                offset_ms,
                source,
            });
        }
    }
    let ordered: Vec<_> = requests
        .iter()
        .map(|r| (&r.source["source_id"], &r.prompt, r.max_tokens))
        .collect();
    let arrivals: Vec<_> = requests.iter().map(|r| r.offset_ms).collect();
    let ordered_hash = super::io::json_hash_bounded(&ordered, recipe.limits.max_output_bytes)?;
    let arrival_hash = super::io::json_hash_bounded(&arrivals, recipe.limits.max_output_bytes)?;
    Ok(PreparedStage {
        client: ClientConfig {
            base_url: recipe.client.base_url.clone(),
            model: recipe.client.model.clone(),
            output_dir: stage.output_dir.clone(),
            requests,
            deadline_ms: recipe.client.deadline_ms,
            temperature: recipe.client.request_policy.sampling.temperature,
            protocol: "chat_completions",
            request_policy: recipe.client.request_policy.clone(),
            provenance: json!({"kind":"synthetic_public_geometry",
                "recipe_sha256":recipe_hash,"ordered_input_sha256":ordered_hash,"arrival_offsets_sha256":arrival_hash}),
        },
        audit: StageAudit {
            schema_version: 1,
            kind: "synthetic_public_geometry",
            model: recipe.client.model.clone(),
            stage: stage.name.clone(),
            context_tokens: recipe.context_tokens,
            kv_block_tokens: recipe.kv_block_tokens,
            requests: audits,
            ordered_input_sha256: ordered_hash,
            arrival_offsets_sha256: arrival_hash,
        },
    })
}
