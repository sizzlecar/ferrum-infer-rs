//! Bounded, signed, version-anchored residual calibration.
//!
//! An empirical quantile within a workload bucket is an approximation. It does
//! not certify conditional coverage or joint, multi-step client latency.

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

mod persistence;
pub(crate) use persistence::{bounded_sequence, PersistedCalibration};

#[derive(Debug, Clone, Copy)]
pub(crate) struct CalibrationConfig {
    pub max_domains: usize,
    pub buckets_per_domain: usize,
    pub window_samples: usize,
    pub min_samples: usize,
    pub ttl: Duration,
    pub quantile: f64,
}

impl CalibrationConfig {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.max_domains == 0
            || self.buckets_per_domain == 0
            || self.min_samples == 0
            || self.min_samples > self.window_samples
            || self.ttl.is_zero()
            || !self.quantile.is_finite()
            || self.quantile <= 0.0
            || self.quantile >= 1.0
        {
            return Err("invalid calibration bounds or quantile");
        }
        self.max_domains
            .checked_mul(
                self.buckets_per_domain
                    .checked_add(1)
                    .ok_or("calibration bucket bound overflow")?,
            )
            .and_then(|n| n.checked_mul(self.window_samples))
            .and_then(|n| n.checked_mul(std::mem::size_of::<Residual>()))
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or("calibration storage bound overflow")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Residual {
    seconds: f64,
    observed_at: Instant,
    ordinal: u64,
}

#[derive(Debug, Clone, Default)]
struct Window(VecDeque<Residual>);

impl Window {
    fn new(config: CalibrationConfig) -> Self {
        Self(VecDeque::with_capacity(config.window_samples))
    }

    fn push(&mut self, sample: Residual, config: CalibrationConfig) {
        if self.0.len() == config.window_samples {
            self.0.pop_front();
        }
        self.0.push_back(sample);
    }

    fn snapshot(&self, now: Instant, config: CalibrationConfig) -> WindowSnapshot {
        // Sort once at publication. Filtering a sorted subset preserves order,
        // so later queries can expire individual samples without allocating.
        let mut samples = Vec::with_capacity(self.0.len());
        samples.extend(self.0.iter().copied().filter(|sample| {
            now.checked_duration_since(sample.observed_at)
                .is_some_and(|age| age < config.ttl)
        }));
        samples.sort_unstable_by(|a: &Residual, b| a.seconds.total_cmp(&b.seconds));
        WindowSnapshot { samples }
    }
}

#[derive(Debug, Clone)]
struct DomainWindow<D> {
    domain: D,
    global: Window,
    buckets: Vec<(u32, Window)>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Correction {
    pub signed_seconds: f64,
    pub samples: usize,
    pub observed_at: Instant,
    pub expires_at: Instant,
    pub conditional: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct CalibrationSnapshot<D> {
    pub model_revision: u64,
    config: CalibrationConfig,
    domains: Vec<(D, WindowSnapshot, Vec<(u32, WindowSnapshot)>)>,
}

#[derive(Debug, Clone)]
struct WindowSnapshot {
    samples: Vec<Residual>,
}

impl WindowSnapshot {
    fn fresh(&self, now: Instant, ttl: Duration) -> impl Iterator<Item = &Residual> {
        self.samples.iter().filter(move |sample| {
            now.checked_duration_since(sample.observed_at)
                .is_some_and(|age| age < ttl)
        })
    }

    fn correction(
        &self,
        now: Instant,
        config: CalibrationConfig,
        quantile: f64,
        conditional: bool,
    ) -> Option<Correction> {
        let mut samples = 0;
        let mut oldest = None;
        let mut newest = None;
        for residual in self.fresh(now, config.ttl) {
            samples += 1;
            oldest = Some(oldest.map_or(residual.observed_at, |old: Instant| {
                old.min(residual.observed_at)
            }));
            newest = Some(newest.map_or(residual.observed_at, |old: Instant| {
                old.max(residual.observed_at)
            }));
        }
        if samples < config.min_samples {
            return None;
        }
        let index = ((quantile * samples as f64).ceil() as usize)
            .saturating_sub(1)
            .min(samples - 1);
        Some(Correction {
            signed_seconds: self.fresh(now, config.ttl).nth(index)?.seconds,
            samples,
            observed_at: newest?,
            expires_at: oldest?.checked_add(config.ttl)?,
            conditional,
        })
    }
}

pub(crate) struct ResidualScenarios {
    pub signed_seconds: Vec<f64>,
    pub correction: Correction,
}

impl<D: PartialEq> CalibrationSnapshot<D> {
    pub fn correction(
        &self,
        revision: u64,
        domain: &D,
        bucket: u32,
        now: Instant,
    ) -> Option<Correction> {
        if revision != self.model_revision {
            return None;
        }
        self.selected(domain, bucket, now)
            .map(|(_, correction)| correction)
    }

    fn selected(
        &self,
        domain: &D,
        bucket: u32,
        now: Instant,
    ) -> Option<(&WindowSnapshot, Correction)> {
        let (_, global, buckets) = self.domains.iter().find(|(key, _, _)| key == domain)?;
        if let Some((_, window)) = buckets.iter().find(|(key, _)| *key == bucket) {
            if let Some(correction) =
                window.correction(now, self.config, self.config.quantile, true)
            {
                return Some((window, correction));
            }
        }
        global
            .correction(now, self.config, self.config.quantile, false)
            .map(|correction| (global, correction))
    }

    /// Complete equally weighted rank bins, represented by their upper empirical
    /// endpoint. Shared ranks across invocations are an explicit dependence
    /// assumption, not a joint coverage guarantee or measured gain interval.
    pub fn scenarios(
        &self,
        revision: u64,
        domain: &D,
        bucket: u32,
        now: Instant,
        count: usize,
    ) -> Option<ResidualScenarios> {
        if revision != self.model_revision || count == 0 || count > 32 {
            return None;
        }
        let (window, correction) = self.selected(domain, bucket, now)?;
        let mut signed_seconds = Vec::with_capacity(count);
        let mut values = window.fresh(now, self.config.ttl).enumerate();
        let mut current = values.next()?;
        for rank in 0..count {
            let index = ((rank + 1) * correction.samples).div_ceil(count) - 1;
            while current.0 < index {
                current = values.next()?;
            }
            signed_seconds.push(current.1.seconds);
        }
        Some(ResidualScenarios {
            signed_seconds,
            correction,
        })
    }
}

#[derive(Debug)]
pub(crate) struct OnlineCalibrator<D> {
    config: CalibrationConfig,
    model_revision: u64,
    domains: Vec<DomainWindow<D>>,
    next_ordinal: u64,
}

impl<D: Clone + PartialEq> OnlineCalibrator<D> {
    /// Requested Rust allocation payload for live state, one published snapshot
    /// and sorting scratch. D must have no separate heap storage for this to
    /// cover its full payload (production uses fixed-size FeatureDomain).
    /// Allocator metadata and snapshots retained by readers are excluded.
    pub fn retained_bytes_bound(config: CalibrationConfig) -> Option<usize> {
        config.validate().ok()?;
        let windows = config.buckets_per_domain.checked_add(1)?;
        let residuals = windows
            .checked_mul(config.window_samples)?
            .checked_mul(2 * std::mem::size_of::<Residual>())?;
        let buckets = config.buckets_per_domain.checked_mul(
            std::mem::size_of::<(u32, Window)>() + std::mem::size_of::<(u32, WindowSnapshot)>(),
        )?;
        let domain_bytes = residuals
            .checked_add(buckets)?
            .checked_add(std::mem::size_of::<DomainWindow<D>>())?
            .checked_add(std::mem::size_of::<(
                D,
                WindowSnapshot,
                Vec<(u32, WindowSnapshot)>,
            )>())?;
        config
            .max_domains
            .checked_mul(domain_bytes)?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(std::mem::size_of::<CalibrationSnapshot<D>>())
    }

    pub fn new(config: CalibrationConfig, model_revision: u64) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self {
            config,
            model_revision,
            domains: Vec::with_capacity(config.max_domains),
            next_ordinal: 0,
        })
    }

    /// The caller predicts before this update. Both actual and risk base refer
    /// to exactly one invocation and the same model revision. Negative errors
    /// are retained, including for a quantile model with an oversized base.
    pub fn observe(
        &mut self,
        revision: u64,
        domain: D,
        bucket: u32,
        actual_seconds: f64,
        risk_base_seconds: f64,
        observed_at: Instant,
    ) -> bool {
        if revision != self.model_revision
            || !actual_seconds.is_finite()
            || actual_seconds <= 0.0
            || !risk_base_seconds.is_finite()
            || risk_base_seconds < 0.0
        {
            return false;
        }
        let residual = actual_seconds - risk_base_seconds;
        if !residual.is_finite() {
            return false;
        }
        let Some(ordinal) = self.next_ordinal.checked_add(1) else {
            return false;
        };
        let index = if let Some(i) = self.domains.iter().position(|d| d.domain == domain) {
            i
        } else {
            if self.domains.len() == self.config.max_domains {
                return false;
            }
            self.domains.push(DomainWindow {
                domain,
                global: Window::new(self.config),
                buckets: Vec::with_capacity(self.config.buckets_per_domain),
            });
            self.domains.len() - 1
        };
        let domain = &mut self.domains[index];
        let sample = Residual {
            seconds: residual,
            observed_at,
            ordinal,
        };
        self.next_ordinal = ordinal;
        domain.global.push(sample, self.config);
        let bucket_index = domain
            .buckets
            .iter()
            .position(|(key, _)| *key == bucket)
            .or_else(|| {
                if domain.buckets.len() == self.config.buckets_per_domain {
                    return None;
                }
                domain.buckets.push((bucket, Window::new(self.config)));
                Some(domain.buckets.len() - 1)
            });
        if let Some(i) = bucket_index {
            domain.buckets[i].1.push(sample, self.config);
        }
        true
    }

    pub fn snapshot(&self, now: Instant) -> CalibrationSnapshot<D> {
        CalibrationSnapshot {
            model_revision: self.model_revision,
            config: self.config,
            domains: self
                .domains
                .iter()
                .map(|d| {
                    let mut buckets = Vec::with_capacity(d.buckets.len());
                    for (key, window) in &d.buckets {
                        buckets.push((*key, window.snapshot(now, self.config)));
                    }
                    (
                        d.domain.clone(),
                        d.global.snapshot(now, self.config),
                        buckets,
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CalibrationConfig {
        CalibrationConfig {
            max_domains: 2,
            buckets_per_domain: 2,
            window_samples: 4,
            min_samples: 2,
            ttl: Duration::from_secs(10),
            quantile: 0.75,
        }
    }

    #[test]
    fn signed_residual_is_anchored_to_the_same_risk_head() {
        let now = Instant::now();
        let mut cal = OnlineCalibrator::new(config(), 7).unwrap();
        assert!(cal.observe(7, 0, 1, 0.08, 0.1, now));
        assert!(cal.observe(7, 0, 1, 0.07, 0.1, now));
        let snapshot = cal.snapshot(now);
        let c = snapshot.correction(7, &0, 1, now).unwrap();
        assert!((0.1 + c.signed_seconds - 0.08).abs() < 1e-12);
        assert!(c.conditional);
        assert!(!cal.observe(8, 0, 1, 0.08, 0.1, now));
        assert!(snapshot.correction(8, &0, 1, now).is_none());
    }

    #[test]
    fn each_bucket_expires_independently_and_falls_back_within_its_domain() {
        let now = Instant::now();
        let later = now + Duration::from_secs(9);
        let mut cal = OnlineCalibrator::new(config(), 1).unwrap();
        for _ in 0..2 {
            cal.observe(1, 0, 1, 2.0, 1.0, now);
        }
        for _ in 0..2 {
            cal.observe(1, 0, 2, 1.5, 1.0, later);
        }
        let expired = now + Duration::from_secs(10);
        let s = cal.snapshot(expired);
        let fallback = s.correction(1, &0, 1, expired).unwrap();
        assert!(!fallback.conditional);
        assert_eq!(fallback.signed_seconds, 0.5);
        assert!(s.correction(1, &0, 2, expired).unwrap().conditional);
        assert!(s.correction(1, &1, 1, expired).is_none());
        assert!(s.correction(1, &0, 2, later + config().ttl).is_none());
    }

    #[test]
    fn newest_observation_does_not_extend_older_qualification() {
        let now = Instant::now();
        let mut cal = OnlineCalibrator::new(config(), 1).unwrap();
        cal.observe(1, 0, 0, 2.0, 1.0, now);
        cal.observe(1, 0, 0, 2.0, 1.0, now + Duration::from_secs(9));
        let s = cal.snapshot(now + Duration::from_secs(9));
        assert!(s
            .correction(1, &0, 0, now + Duration::from_secs(10))
            .is_none());
    }

    #[test]
    fn windows_and_domains_remain_bounded_without_borrowing_other_domains() {
        let now = Instant::now();
        let mut cal = OnlineCalibrator::new(config(), 1).unwrap();
        for domain in 0..2 {
            for bucket in 0..9 {
                for _ in 0..8 {
                    assert!(cal.observe(1, domain, bucket, 2.0, 1.0, now));
                }
            }
        }
        assert!(!cal.observe(1, 2, 0, 2.0, 1.0, now));
        for domain in &cal.domains {
            assert_eq!(domain.global.0.len(), 4);
            assert_eq!(domain.buckets.len(), 2);
            assert!(domain.buckets.iter().all(|(_, w)| w.0.len() <= 4));
        }
        assert!(!cal.observe(1, 0, 0, f64::NAN, 1.0, now));
        assert!(!cal.observe(1, 0, 0, 1.0, f64::INFINITY, now));
    }

    #[test]
    fn retained_bound_uses_actual_payload_types_and_preallocated_capacities() {
        let config = config();
        let mut cal = OnlineCalibrator::<u8>::new(config, 1).unwrap();
        let now = Instant::now();
        for domain in 0..config.max_domains as u8 {
            for bucket in 0..config.buckets_per_domain as u32 {
                for _ in 0..config.window_samples {
                    cal.observe(1, domain, bucket, 1.0, 0.5, now);
                }
            }
        }
        let snapshot = cal.snapshot(now);
        let mut payload = std::mem::size_of_val(&cal)
            + std::mem::size_of_val(&snapshot)
            + cal.domains.capacity() * std::mem::size_of::<DomainWindow<u8>>()
            + snapshot.domains.capacity()
                * std::mem::size_of::<(u8, WindowSnapshot, Vec<(u32, WindowSnapshot)>)>();
        for domain in &cal.domains {
            payload += domain.global.0.capacity() * std::mem::size_of::<Residual>();
            payload += domain.buckets.capacity() * std::mem::size_of::<(u32, Window)>();
            for (_, window) in &domain.buckets {
                payload += window.0.capacity() * std::mem::size_of::<Residual>();
            }
        }
        for (_, global, buckets) in &snapshot.domains {
            payload += global.samples.capacity() * std::mem::size_of::<Residual>();
            payload += buckets.capacity() * std::mem::size_of::<(u32, WindowSnapshot)>();
            for (_, window) in buckets {
                payload += window.samples.capacity() * std::mem::size_of::<Residual>();
            }
        }
        assert!(payload <= OnlineCalibrator::<u8>::retained_bytes_bound(config).unwrap());
        assert!(
            OnlineCalibrator::<u8>::retained_bytes_bound(CalibrationConfig {
                window_samples: usize::MAX,
                ..config
            })
            .is_none()
        );
    }

    #[test]
    fn immutable_snapshot_expires_old_outlier_without_discarding_fresh_qualification() {
        let now = Instant::now();
        let later = now + Duration::from_secs(9);
        let mut cal = OnlineCalibrator::new(config(), 1).unwrap();
        cal.observe(1, 0, 0, 11.0, 1.0, now);
        cal.observe(1, 0, 0, 1.1, 1.0, later);
        cal.observe(1, 0, 0, 1.2, 1.0, later);
        let snapshot = cal.snapshot(later);
        assert_eq!(
            snapshot.correction(1, &0, 0, later).unwrap().signed_seconds,
            10.0
        );
        let fresh = snapshot
            .correction(1, &0, 0, now + Duration::from_secs(11))
            .unwrap();
        assert_eq!(fresh.samples, 2);
        assert!((fresh.signed_seconds - 0.2).abs() < 1e-12);
        assert_eq!(fresh.expires_at, now + Duration::from_secs(19));
        assert!(snapshot
            .correction(1, &0, 0, now + Duration::from_secs(19))
            .is_none());
    }

    #[test]
    fn empirical_scenario_bins_keep_signed_mass_and_filter_expired_values() {
        let now = Instant::now();
        let later = now + Duration::from_secs(9);
        let mut cal = OnlineCalibrator::new(config(), 1).unwrap();
        cal.observe(1, 0, 0, 10.0, 1.0, now);
        cal.observe(1, 0, 0, 0.5, 1.0, later);
        cal.observe(1, 0, 0, 1.5, 1.0, later);
        let snapshot = cal.snapshot(later);
        let bins = snapshot
            .scenarios(1, &0, 0, now + Duration::from_secs(11), 4)
            .unwrap();
        assert_eq!(bins.signed_seconds, vec![-0.5, -0.5, 0.5, 0.5]);
        assert!(snapshot.scenarios(2, &0, 0, later, 4).is_none());
        assert!(snapshot.scenarios(1, &0, 0, later, 33).is_none());
    }
}
