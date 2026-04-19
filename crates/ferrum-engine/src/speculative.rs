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

use ferrum_types::{Result, TokenId};
use rand::RngCore;

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

#[cfg(test)]
mod tests {
    use super::*;
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
