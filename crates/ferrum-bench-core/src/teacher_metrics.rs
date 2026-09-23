//! Full-vocabulary numerical metrics for identical teacher histories.
//! Artifact identity, history alignment and quality budgets are checked by
//! callers; these metrics alone do not certify a model or a release.
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TeacherNll {
    pub token_id: u32,
    pub reference_nll_nats: f64,
    pub candidate_nll_nats: f64,
    pub delta_nll_nats: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FullVocabularyMetrics {
    pub vocabulary_size: usize,
    pub kl_reference_to_candidate_nats: f64,
    pub reference_argmax: usize,
    pub candidate_argmax: usize,
    pub argmax_agrees: bool,
    pub nmse: f64,
    pub max_abs: f64,
    pub teacher_forced: Option<TeacherNll>,
}

/// Compensated accumulation shared by per-decision metrics and aggregation.
pub fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut sum, mut correction) = (0.0, 0.0);
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn log_probabilities(logits: &[f32]) -> Vec<f64> {
    // Subtract in F64 before taking the partition logarithm. This preserves
    // small log probabilities even when all logits have a large common offset.
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_partition =
        compensated_sum(logits.iter().map(|&x| (f64::from(x) - maximum).exp())).ln();
    logits
        .iter()
        .map(|&x| (f64::from(x) - maximum) - log_partition)
        .collect()
}

/// Compare complete finite raw logits. Ties select the lowest token index.
pub fn compare_full_vocabulary(
    reference: &[f32],
    candidate: &[f32],
    teacher_token: Option<u32>,
) -> Result<FullVocabularyMetrics, String> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return Err("teacher distributions require equal nonempty vocabularies".into());
    }
    if reference.iter().chain(candidate).any(|x| !x.is_finite()) {
        return Err("teacher distributions contain non-finite logits".into());
    }
    let target = teacher_token
        .map(|token| {
            usize::try_from(token)
                .ok()
                .filter(|&index| index < reference.len())
                .ok_or_else(|| "teacher-forced token is outside checkpoint vocabulary".to_owned())
        })
        .transpose()?;
    let p = log_probabilities(reference);
    let q = log_probabilities(candidate);
    // This is the entire vocabulary, not a top-k or argmax-only comparison.
    let kl = compensated_sum(p.iter().zip(&q).map(|(&p, &q)| p.exp() * (p - q))).max(0.0);
    let teacher_forced = target.map(|token| TeacherNll {
        token_id: teacher_token.expect("validated target"),
        reference_nll_nats: -p[token],
        candidate_nll_nats: -q[token],
        delta_nll_nats: p[token] - q[token],
    });
    let argmax = |values: &[f32]| {
        (1..values.len()).fold(0, |best, index| {
            if values[index] > values[best] {
                index
            } else {
                best
            }
        })
    };
    let reference_argmax = argmax(reference);
    let candidate_argmax = argmax(candidate);
    Ok(FullVocabularyMetrics {
        vocabulary_size: reference.len(),
        kl_reference_to_candidate_nats: kl,
        reference_argmax,
        candidate_argmax,
        argmax_agrees: reference_argmax == candidate_argmax,
        nmse: crate::release_regression::numerics::nmse(reference, candidate),
        max_abs: reference
            .iter()
            .zip(candidate)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).abs())
            .fold(0.0, f64::max),
        teacher_forced,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_vocabulary_rejects_missing_or_nonfinite_data() {
        for (a, b) in [
            (&[][..], &[][..]),
            (&[1.0][..], &[1.0, 2.0][..]),
            (&[f32::NAN][..], &[0.0][..]),
            (&[0.0][..], &[f32::INFINITY][..]),
        ] {
            assert!(compare_full_vocabulary(a, b, None).is_err());
        }
        assert!(compare_full_vocabulary(&[0.0], &[0.0], Some(1)).is_err());
    }

    #[test]
    fn complete_vocabulary_measures_tail_even_when_argmax_agrees() {
        let p = [0.0, -1.0, -2.0, -3.0];
        let q = [0.0, -1.0, -2.0, -6.0];
        let m = compare_full_vocabulary(&p, &q, Some(3)).unwrap();
        assert!(m.argmax_agrees);
        assert!(m.kl_reference_to_candidate_nats > 0.0);
        assert!(m.teacher_forced.unwrap().delta_nll_nats > 2.0);
    }
}
