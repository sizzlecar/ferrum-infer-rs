use super::*;

/// Select a conservative one-based tail rank using the binomial lower-tail
/// Chernoff bound. If X counts draws beyond a true bootstrap tail quantile,
/// E[X] >= B*p and P(X < k) <= exp(-(B*p-k)^2/(2*B*p)). Taking one rank farther
/// outward than floor(B*p-sqrt(2*B*p*log(1/delta))) avoids optimistic rounding.
/// This controls simulation error conditional on the empirical distribution;
/// it does NOT validate the bootstrap's population coverage approximation.
pub(super) fn tail_rank(resamples: usize, tail: f64, failure: f64) -> Option<usize> {
    if resamples == 0
        || !tail.is_finite()
        || !(0.0..1.0).contains(&tail)
        || tail == 0.0
        || !failure.is_finite()
        || !(0.0..1.0).contains(&failure)
        || failure == 0.0
    {
        return None;
    }
    let expected = resamples as f64 * tail;
    let margin = (2.0 * expected * -failure.ln()).sqrt();
    let rank = (expected - margin).floor() - 1.0;
    (rank >= 1.0 && rank <= resamples as f64).then_some(rank as usize)
}

/// Versioned SHA-256 counter stream with unbiased rejection mapping. A single
/// index draw selects the complete vector of primary cells and metrics. No
/// rand implementation detail or traversal of an input HashMap affects it.
struct CounterDraws {
    seed: u64,
    counter: u64,
    bytes: [u8; 32],
    offset: usize,
    remaining_words: usize,
}

impl CounterDraws {
    fn new(seed: u64, draws: usize) -> Option<Self> {
        Some(Self {
            seed,
            counter: 0,
            bytes: [0; 32],
            offset: 32,
            remaining_words: draws.checked_mul(2)?.checked_add(64)?,
        })
    }

    fn index(&mut self, count: usize) -> Option<usize> {
        let count = u64::try_from(count).ok().filter(|count| *count != 0)?;
        let limit = u64::MAX - u64::MAX % count;
        loop {
            self.remaining_words = self.remaining_words.checked_sub(1)?;
            if self.offset == self.bytes.len() {
                let mut hasher = Sha256::new();
                hasher.update(b"ferrum-paired-cluster-bootstrap-v1\0");
                hasher.update(self.seed.to_le_bytes());
                hasher.update(self.counter.to_le_bytes());
                self.bytes.copy_from_slice(&hasher.finalize());
                self.counter = self.counter.checked_add(1)?;
                self.offset = 0;
            }
            let value =
                u64::from_le_bytes(self.bytes[self.offset..self.offset + 8].try_into().ok()?);
            self.offset += 8;
            if value < limit {
                return usize::try_from(value % count).ok();
            }
        }
    }
}

/// Input columns are complete paired-ratio series in frozen primary-cell,
/// metric and pair order. Never impute, trim or independently resample columns.
pub(super) fn resample_means(
    columns: &[Vec<f64>],
    resamples: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, ComparisonError> {
    let pairs = columns.first().map_or(0, Vec::len);
    resample_means_for_size(columns, resamples, pairs, seed)
}

/// Pilot precision planning samples the empirical paired population at a
/// preregistered prospective allocation; production intervals use N=observed N.
pub(super) fn resample_means_for_size(
    columns: &[Vec<f64>],
    resamples: usize,
    sampled_pairs: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, ComparisonError> {
    let pairs = columns.first().map_or(0, Vec::len);
    if columns.len() % ComparisonMetric::ALL.len() != 0 {
        return Err(ComparisonError(
            "incomplete seven-metric primary family".into(),
        ));
    }
    configuration::check_work(
        sampled_pairs,
        resamples,
        columns.len() / ComparisonMetric::ALL.len(),
    )?;
    configuration::check_work(pairs, 1, columns.len() / ComparisonMetric::ALL.len())?;
    if columns.iter().any(|column| {
        column.len() != pairs
            || column
                .iter()
                .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
    }) {
        return Err(ComparisonError(
            "bootstrap needs complete positive finite paired ratios".into(),
        ));
    }
    let draws = sampled_pairs
        .checked_mul(resamples)
        .ok_or_else(|| ComparisonError("bootstrap draw overflow".into()))?;
    let mut rng = CounterDraws::new(seed, draws)
        .ok_or_else(|| ComparisonError("bootstrap draw budget overflow".into()))?;
    let mut distributions: Vec<Vec<f64>> = (0..columns.len())
        .map(|_| Vec::with_capacity(resamples))
        .collect();
    let mut sums = vec![0.0; columns.len()];
    for _ in 0..resamples {
        sums.fill(0.0);
        for _ in 0..sampled_pairs {
            let index = rng
                .index(pairs)
                .ok_or_else(|| ComparisonError("bootstrap random draw budget exhausted".into()))?;
            for (sum, column) in sums.iter_mut().zip(columns) {
                *sum += column[index] / sampled_pairs as f64;
            }
        }
        for (distribution, sum) in distributions.iter_mut().zip(&sums) {
            if !sum.is_finite() || *sum <= 0.0 {
                return Err(ComparisonError(
                    "bootstrap mean is zero or non-finite".into(),
                ));
            }
            distribution.push(*sum);
        }
    }
    Ok(distributions)
}
