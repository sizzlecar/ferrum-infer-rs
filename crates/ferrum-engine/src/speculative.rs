//! Speculative decoding — draft + verify.
//!
//! A small ("draft") model generates `N` candidate tokens autoregressively.
//! The big ("target") model does ONE forward pass on prompt + drafts and
//! produces `N+1` logit distributions — one for every position at which a
//! draft could be accepted, plus one bonus position after full acceptance.
//! We then apply the DeepMind speculative sampling rule to decide which
//! drafts survive, yielding 1..=N+1 tokens per target forward pass.
//!
//! Original paper: Leviathan et al., 2023 — "Fast Inference from Transformers
//! via Speculative Decoding" (https://arxiv.org/abs/2211.17192).
//!
//! This module is **algorithm-only**: it operates on raw logit vectors and
//! produces a list of accepted token ids. The engine/scheduler integration
//! (draft-model loading, KV-cache management, iteration plumbing) is a
//! separate layer and explicitly out of scope for this file — wiring it is
//! the follow-up once the algorithm is locked down.

use ferrum_interfaces::{
    model_executor::{DecodeInput, DecodeOutput},
    tensor::TensorFactory,
    KvCacheHandle, ModelExecutor,
};
use ferrum_types::{Result, TokenId};
use rand::RngCore;
use std::sync::Arc;

/// Softmax a logit vector, in-place modifications avoided — returns new Vec.
/// `temperature == 0.0` collapses to one-hot at argmax (as elsewhere in the
/// sampler stack).
fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    if temperature == 0.0 {
        // Greedy: delta at argmax.
        let (argmax, _) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            });
        let mut p = vec![0.0f32; logits.len()];
        p[argmax] = 1.0;
        return p;
    }
    let inv_t = 1.0 / temperature;
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    if !max.is_finite() {
        max = 0.0;
    }
    let mut sum = 0.0f64;
    let mut out = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = ((l - max) * inv_t).exp();
        out.push(e);
        sum += e as f64;
    }
    let inv_sum = (1.0 / sum) as f32;
    for x in out.iter_mut() {
        *x *= inv_sum;
    }
    out
}

/// Sample a token id from a probability distribution using uniform `u` in [0, 1).
fn sample_categorical(probs: &[f32], u: f32) -> TokenId {
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if u <= acc {
            return TokenId::new(i as u32);
        }
    }
    TokenId::new((probs.len().saturating_sub(1)) as u32)
}

/// Draw a fresh `u ~ U[0,1)` from any RngCore.
fn next_u(rng: &mut dyn RngCore) -> f32 {
    // Convert a u32 to f32 in [0, 1): divide by 2^32.
    (rng.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
}

/// Residual distribution (p_T - p_D) clipped at zero and renormalised.
/// This is the distribution we sample from when a draft is rejected.
fn residual(p_target: &[f32], p_draft: &[f32]) -> Vec<f32> {
    debug_assert_eq!(p_target.len(), p_draft.len());
    let mut r = Vec::with_capacity(p_target.len());
    let mut sum = 0.0f64;
    for (&pt, &pd) in p_target.iter().zip(p_draft.iter()) {
        let d = (pt - pd).max(0.0);
        r.push(d);
        sum += d as f64;
    }
    if sum <= 0.0 {
        // Shouldn't happen unless the target and draft fully agree on the
        // draft token (probability 1, which the accept rule already catches).
        // Fall back to the target distribution.
        return p_target.to_vec();
    }
    let inv = (1.0 / sum) as f32;
    for x in r.iter_mut() {
        *x *= inv;
    }
    r
}

/// Input to `verify_speculation`: logit vectors at each speculation position.
///
/// For a speculation of `N` draft tokens:
/// - `draft_logits.len() == N` — the distribution the draft model used to
///   sample each draft token (aligned with `draft_tokens`).
/// - `target_logits.len() == N + 1` — the target model's distributions at
///   each position; the extra slot is the bonus-token draw after all drafts
///   are accepted.
/// - `draft_tokens.len() == N`.
pub struct Speculation<'a> {
    pub draft_tokens: &'a [TokenId],
    pub draft_logits: &'a [Vec<f32>],
    pub target_logits: &'a [Vec<f32>],
    pub temperature: f32,
}

/// Result of one speculate+verify round.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculationOutcome {
    /// Tokens that survived verification — always at least 1 (either the
    /// residual resample on a rejection or the bonus token on full accept).
    pub tokens: Vec<TokenId>,
    /// Index of the first rejected draft (0..N for rejection, N for full
    /// accept). Useful for KV-cache rollback in the engine.
    pub rejected_at: usize,
}

/// Execute the DeepMind speculative-sampling accept/reject loop.
///
/// Per-draft decision (draft i, i = 0..N):
///   - Let p_T = target prob of draft[i] at position i.
///   - Let p_D = draft prob of draft[i] at position i.
///   - Accept with probability `min(1, p_T / p_D)`.
///   - If rejected, sample a replacement from the residual `(p_T - p_D)+`.
///
/// If every draft is accepted, sample a bonus token from `target_logits[N]`.
pub fn verify_speculation(spec: Speculation<'_>, rng: &mut dyn RngCore) -> Result<SpeculationOutcome> {
    let n = spec.draft_tokens.len();
    assert_eq!(spec.draft_logits.len(), n, "draft_logits count mismatch");
    assert_eq!(
        spec.target_logits.len(),
        n + 1,
        "target_logits must have N+1 rows (positions 0..N)"
    );

    let mut accepted: Vec<TokenId> = Vec::with_capacity(n + 1);

    for i in 0..n {
        let draft_token = spec.draft_tokens[i];
        let idx = draft_token.get() as usize;

        let p_target = softmax(&spec.target_logits[i], spec.temperature);
        let p_draft = softmax(&spec.draft_logits[i], spec.temperature);

        if idx >= p_target.len() || idx >= p_draft.len() {
            // Malformed input — fall back to a greedy target pick and stop.
            let t = TokenId::new(
                p_target
                    .iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |(bi, bv), (j, &v)| {
                        if v > bv {
                            (j, v)
                        } else {
                            (bi, bv)
                        }
                    })
                    .0 as u32,
            );
            accepted.push(t);
            return Ok(SpeculationOutcome {
                tokens: accepted,
                rejected_at: i,
            });
        }

        let pt = p_target[idx];
        let pd = p_draft[idx].max(1e-20); // avoid division by zero
        let ratio = (pt / pd).min(1.0);
        let u = next_u(rng);
        if u < ratio {
            // Accepted.
            accepted.push(draft_token);
        } else {
            // Rejected: sample from residual (p_T - p_D)+.
            let res = residual(&p_target, &p_draft);
            let replacement = sample_categorical(&res, next_u(rng));
            accepted.push(replacement);
            return Ok(SpeculationOutcome {
                tokens: accepted,
                rejected_at: i,
            });
        }
    }

    // All drafts accepted — sample a bonus token from the target's trailing
    // distribution so the round always produces at least one NEW token even
    // if the draft happened to end exactly where generation should stop.
    let p_bonus = softmax(&spec.target_logits[n], spec.temperature);
    let bonus = sample_categorical(&p_bonus, next_u(rng));
    accepted.push(bonus);
    Ok(SpeculationOutcome {
        tokens: accepted,
        rejected_at: n,
    })
}

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeDecodingConfig {
    /// Draft model produces this many tokens per target forward pass.
    /// Typical range: 3-7. Paper used 4-5. Larger N amortises target cost
    /// more aggressively but raises the probability of early rejection.
    pub num_speculative_tokens: usize,
    /// Temperature applied during accept/reject. Matches the sampling
    /// temperature to keep the target-draft ratio calibrated.
    pub temperature: f32,
}

impl Default for SpeculativeDecodingConfig {
    fn default() -> Self {
        Self {
            num_speculative_tokens: 4,
            temperature: 1.0,
        }
    }
}

/// Drives one round of speculative decoding against a (draft, target) pair
/// of `ModelExecutor`s. Owns neither executor — the engine keeps them and
/// hands references per call.
///
/// Usage per decode iteration:
///   - Caller hands the runner the last sampled token for this request,
///     plus the draft & target KV cache handles.
///   - `step()` runs N draft decodes, N+1 target decodes (sequentially for
///     now — performance gain lands once the executors grow multi-position
///     decode support), then applies `verify_speculation`.
///   - Returns the list of newly accepted tokens plus the updated KV
///     handles for draft and target (caller installs them on the sequence
///     state for the next iteration).
pub struct SpeculativeRunner<'a> {
    pub draft: &'a dyn ModelExecutor,
    pub target: &'a dyn ModelExecutor,
    pub tensor_factory: Arc<dyn TensorFactory>,
    pub cfg: SpeculativeDecodingConfig,
}

/// Result of a single `SpeculativeRunner::step`.
pub struct SpeculativeStepOutcome {
    pub tokens: Vec<TokenId>,
    pub draft_kv: Arc<dyn KvCacheHandle>,
    pub target_kv: Arc<dyn KvCacheHandle>,
    /// True when a draft was rejected (caller may want to roll target KV
    /// back to `rejected_at` to stay token-aligned with the draft model).
    pub rejected: bool,
    pub rejected_at: usize,
    /// Draft KV is `N` writes; target KV is `N+1` writes. In the full-accept
    /// case the draft is exactly one position behind target. Caller
    /// catches up by feeding this token (the final draft input to target,
    /// i.e. `draft_tokens[N-1]`) into the draft executor once. `None` on
    /// rejection (caller handles rollback separately).
    pub draft_catchup_token: Option<TokenId>,
}

impl<'a> SpeculativeRunner<'a> {
    /// Run one draft+verify cycle. `last_token` is the token that was sampled
    /// from the previous iteration (or the end of prefill) — it's the
    /// starting point both executors advance from.
    pub async fn step(
        &self,
        last_token: TokenId,
        draft_kv: Arc<dyn KvCacheHandle>,
        target_kv: Arc<dyn KvCacheHandle>,
        rng: &mut (dyn RngCore + Send),
    ) -> Result<SpeculativeStepOutcome> {
        let n = self.cfg.num_speculative_tokens.max(1);

        // ── Draft: N sequential decodes, one token at a time ─────────────
        let mut draft_tokens: Vec<TokenId> = Vec::with_capacity(n);
        let mut draft_logits: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut draft_kv_cur = draft_kv;
        let mut draft_prev_token = last_token;
        for _ in 0..n {
            let input_tensor = tokens_to_tensor(&self.tensor_factory, &[draft_prev_token.get()])?;
            let input = DecodeInput::new(input_tensor, draft_kv_cur.clone());
            let output = self.draft.decode(&input).await?;
            let logits = output.logits.to_vec_f32()?;
            let next_token = argmax_token(&logits);
            draft_tokens.push(next_token);
            draft_logits.push(logits);
            draft_kv_cur = output.kv_cache.clone();
            draft_prev_token = next_token;
        }

        // ── Target: ONE multi-position forward over N+1 tokens ──────────
        // Feeds [last_token, draft_0, ..., draft_{N-1}] into a single
        // forward pass and gets N+1 logit rows back — one per position.
        // Dramatically cheaper than N+1 sequential decodes because the
        // weight matrices are read from HBM exactly once instead of N+1
        // times (the decode hot path is memory-bound).
        let mut verify_tokens = Vec::with_capacity(n + 1);
        verify_tokens.push(last_token);
        for i in 0..n {
            verify_tokens.push(draft_tokens[i]);
        }
        let mut verify_inputs = Vec::with_capacity(verify_tokens.len());
        let mut kv_for_verify = target_kv.clone();
        for tok in &verify_tokens {
            let input_tensor = tokens_to_tensor(&self.tensor_factory, &[tok.get()])?;
            verify_inputs.push(DecodeInput::new(input_tensor, kv_for_verify.clone()));
            // Subsequent inputs share the same handle; `forward_verify` only
            // reads cache_id + starting seq from the first one.
            kv_for_verify = kv_for_verify.clone();
        }
        let verify_outputs: Vec<DecodeOutput> =
            self.target.forward_verify(&verify_inputs).await?;
        assert_eq!(verify_outputs.len(), n + 1);
        let mut target_logits: Vec<Vec<f32>> = Vec::with_capacity(n + 1);
        for out in &verify_outputs {
            target_logits.push(out.logits.to_vec_f32()?);
        }
        let target_kv_cur = verify_outputs
            .last()
            .map(|o| o.kv_cache.clone())
            .unwrap_or(target_kv);

        let spec = Speculation {
            draft_tokens: &draft_tokens,
            draft_logits: &draft_logits,
            target_logits: &target_logits,
            temperature: self.cfg.temperature,
        };
        let outcome = verify_speculation(spec, rng)?;
        let rejected = outcome.rejected_at < n;
        // For full-accept, the token needed to realign draft with target is
        // the last draft input consumed by target: draft_tokens[N-1].
        let draft_catchup_token = if !rejected {
            draft_tokens.last().copied()
        } else {
            None
        };
        Ok(SpeculativeStepOutcome {
            tokens: outcome.tokens,
            draft_kv: draft_kv_cur,
            target_kv: target_kv_cur,
            rejected,
            rejected_at: outcome.rejected_at,
            draft_catchup_token,
        })
    }
}

fn tokens_to_tensor(
    factory: &Arc<dyn TensorFactory>,
    token_ids: &[u32],
) -> Result<ferrum_interfaces::tensor::TensorRef> {
    use ferrum_types::{DataType, Device};
    let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
    let len = f32_data.len();
    factory.from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
}

fn argmax_token(logits: &[f32]) -> TokenId {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        });
    TokenId::new(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::KvCacheHandle;
    use ferrum_testkit::{ConfigurableModelExecutor, MockKvCacheHandle, MockTensorFactory};
    use ferrum_types::RequestId;
    use rand::{rngs::StdRng, SeedableRng};

    /// Build a logit vector biased toward `favored` with `strength` over a
    /// `vocab_size`-entry uniform baseline.
    fn biased_logits(vocab_size: usize, favored: u32, strength: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab_size];
        if (favored as usize) < vocab_size {
            v[favored as usize] = strength;
        }
        v
    }

    /// Greedy agreement: draft picks the same token as target → full accept
    /// plus bonus, regardless of rng.
    #[test]
    fn full_accept_when_draft_matches_target() {
        let vocab = 32;
        let drafts = vec![TokenId::new(3), TokenId::new(7), TokenId::new(11)];
        let dl: Vec<Vec<f32>> = drafts
            .iter()
            .map(|t| biased_logits(vocab, t.get(), 20.0))
            .collect();
        // Target agrees perfectly on positions 0..2 and at the bonus slot 3.
        let tl: Vec<Vec<f32>> = drafts
            .iter()
            .map(|t| biased_logits(vocab, t.get(), 20.0))
            .chain(std::iter::once(biased_logits(vocab, 19, 20.0)))
            .collect();

        let mut rng = StdRng::seed_from_u64(1);
        let out = verify_speculation(
            Speculation {
                draft_tokens: &drafts,
                draft_logits: &dl,
                target_logits: &tl,
                temperature: 1.0,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(out.rejected_at, 3, "no rejections → rejected_at == N");
        assert_eq!(
            out.tokens,
            vec![TokenId::new(3), TokenId::new(7), TokenId::new(11), TokenId::new(19)],
            "should accept all three drafts + sample the bonus (token 19)"
        );
    }

    /// First-draft rejection: draft puts mass on token A, target puts ~0 on
    /// A and heavy on B → ratio ≈ 0 → reject → residual resamples toward B.
    #[test]
    fn first_draft_rejected_residual_prefers_target() {
        let vocab = 16;
        let drafts = vec![TokenId::new(2)];
        let dl = vec![biased_logits(vocab, 2, 20.0)];
        // Target says token 5 is the right answer, not token 2.
        let tl = vec![
            biased_logits(vocab, 5, 20.0),
            biased_logits(vocab, 0, 0.0), // unused — rejection stops the loop
        ];

        let mut rng = StdRng::seed_from_u64(7);
        let out = verify_speculation(
            Speculation {
                draft_tokens: &drafts,
                draft_logits: &dl,
                target_logits: &tl,
                temperature: 1.0,
            },
            &mut rng,
        )
        .unwrap();

        assert_eq!(out.rejected_at, 0);
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(
            out.tokens[0],
            TokenId::new(5),
            "residual should pick target's preferred token"
        );
    }

    /// Partial acceptance: first draft matches target, second doesn't.
    /// Expect exactly 2 tokens out: accepted[0] + residual replacement.
    #[test]
    fn partial_acceptance_second_draft_rejected() {
        let vocab = 16;
        let drafts = vec![TokenId::new(4), TokenId::new(9)];
        let dl = vec![
            biased_logits(vocab, 4, 20.0), // draft 0: prefers token 4
            biased_logits(vocab, 9, 20.0), // draft 1: prefers token 9
        ];
        let tl = vec![
            biased_logits(vocab, 4, 20.0), // target agrees at pos 0
            biased_logits(vocab, 1, 20.0), // target disagrees at pos 1 (wants 1)
            biased_logits(vocab, 0, 0.0),  // unused
        ];

        let mut rng = StdRng::seed_from_u64(42);
        let out = verify_speculation(
            Speculation {
                draft_tokens: &drafts,
                draft_logits: &dl,
                target_logits: &tl,
                temperature: 1.0,
            },
            &mut rng,
        )
        .unwrap();

        assert_eq!(out.rejected_at, 1);
        assert_eq!(out.tokens.len(), 2);
        assert_eq!(out.tokens[0], TokenId::new(4));
        assert_eq!(
            out.tokens[1],
            TokenId::new(1),
            "replacement should be the target's preferred token at position 1"
        );
    }

    /// Empty speculation (N=0): just draws the target's bonus token.
    #[test]
    fn zero_drafts_returns_bonus_only() {
        let vocab = 8;
        let tl = vec![biased_logits(vocab, 7, 20.0)];
        let mut rng = StdRng::seed_from_u64(0);
        let out = verify_speculation(
            Speculation {
                draft_tokens: &[],
                draft_logits: &[],
                target_logits: &tl,
                temperature: 1.0,
            },
            &mut rng,
        )
        .unwrap();
        assert_eq!(out.rejected_at, 0);
        assert_eq!(out.tokens, vec![TokenId::new(7)]);
    }

    /// Temperature 0 (greedy) with matching argmaxes should behave like the
    /// agreement case — always full accept + bonus argmax.
    #[test]
    fn greedy_temperature_full_accept_deterministic() {
        let vocab = 16;
        let drafts = vec![TokenId::new(2), TokenId::new(5)];
        let dl: Vec<Vec<f32>> = drafts
            .iter()
            .map(|t| biased_logits(vocab, t.get(), 10.0))
            .collect();
        let tl: Vec<Vec<f32>> = drafts
            .iter()
            .map(|t| biased_logits(vocab, t.get(), 10.0))
            .chain(std::iter::once(biased_logits(vocab, 13, 10.0)))
            .collect();

        let mut rng = StdRng::seed_from_u64(999);
        let out = verify_speculation(
            Speculation {
                draft_tokens: &drafts,
                draft_logits: &dl,
                target_logits: &tl,
                temperature: 0.0,
            },
            &mut rng,
        )
        .unwrap();

        assert_eq!(out.rejected_at, 2);
        assert_eq!(
            out.tokens,
            vec![TokenId::new(2), TokenId::new(5), TokenId::new(13)]
        );
    }

    // ── SpeculativeRunner integration tests (mock executors) ─────────

    fn mock_kv(num_layers: usize) -> Arc<dyn KvCacheHandle> {
        Arc::new(MockKvCacheHandle::new(RequestId::new(), num_layers, 0))
    }

    /// Draft and target both use the same executor logic (matching logits)
    /// → every draft should be accepted and a bonus token should follow.
    #[tokio::test]
    async fn runner_full_accept_when_models_agree() {
        let vocab = 64;
        let draft: Arc<ConfigurableModelExecutor> =
            Arc::new(ConfigurableModelExecutor::with_token_sequence(
                vocab,
                vec![13, 13, 13, 13, 13],
            ));
        let target: Arc<ConfigurableModelExecutor> =
            Arc::new(ConfigurableModelExecutor::with_token_sequence(
                vocab,
                vec![13, 13, 13, 13, 13],
            ));
        let tf: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
        let runner = SpeculativeRunner {
            draft: draft.as_ref(),
            target: target.as_ref(),
            tensor_factory: tf,
            cfg: SpeculativeDecodingConfig {
                num_speculative_tokens: 3,
                temperature: 1.0,
            },
        };
        let mut rng = StdRng::seed_from_u64(0);
        let out = runner
            .step(TokenId::new(5), mock_kv(12), mock_kv(12), &mut rng)
            .await
            .expect("step");

        assert!(!out.rejected, "agreeing models should not reject");
        assert_eq!(out.rejected_at, 3);
        assert_eq!(out.tokens.len(), 4, "3 drafts + 1 bonus");
        for &t in &out.tokens {
            assert_eq!(
                t.get(),
                13,
                "agreeing models should all emit the biased token 13"
            );
        }
    }

    /// Draft biases token A, target biases token B → first draft rejected
    /// → step returns 1 replacement token (target's preferred). Remaining
    /// draft positions don't contribute.
    #[tokio::test]
    async fn runner_rejects_when_models_disagree() {
        let vocab = 64;
        let draft = Arc::new(ConfigurableModelExecutor::with_token_sequence(
            vocab,
            vec![7, 7, 7],
        ));
        let target = Arc::new(ConfigurableModelExecutor::with_token_sequence(
            vocab,
            vec![21, 21, 21, 21],
        ));
        let tf: Arc<dyn TensorFactory> = Arc::new(MockTensorFactory);
        let runner = SpeculativeRunner {
            draft: draft.as_ref(),
            target: target.as_ref(),
            tensor_factory: tf,
            cfg: SpeculativeDecodingConfig {
                num_speculative_tokens: 3,
                temperature: 1.0,
            },
        };
        let mut rng = StdRng::seed_from_u64(1);
        let out = runner
            .step(TokenId::new(0), mock_kv(12), mock_kv(12), &mut rng)
            .await
            .expect("step");

        assert!(out.rejected);
        assert_eq!(out.rejected_at, 0, "first draft should be rejected");
        assert_eq!(out.tokens.len(), 1);
        assert_eq!(
            out.tokens[0].get(),
            21,
            "residual should sample target's preferred token"
        );
    }

    /// Sanity: unbiased-vs-unbiased (both distributions uniform) — the
    /// algorithm always accepts because ratio = 1. Ensures no accidental
    /// rejection due to floating-point wobble on equal distributions.
    #[test]
    fn equal_distributions_always_accept() {
        let vocab = 8;
        let drafts = vec![TokenId::new(3)];
        let dl = vec![vec![0.0f32; vocab]];
        let tl = vec![vec![0.0f32; vocab], vec![0.0f32; vocab]];
        for seed in 0..20u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let out = verify_speculation(
                Speculation {
                    draft_tokens: &drafts,
                    draft_logits: &dl,
                    target_logits: &tl,
                    temperature: 1.0,
                },
                &mut rng,
            )
            .unwrap();
            assert_eq!(out.rejected_at, 1, "seed {seed}: should accept");
            assert_eq!(out.tokens.len(), 2);
            assert_eq!(out.tokens[0], TokenId::new(3));
        }
    }
}
