use super::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedResidual {
    seconds: f64,
    age_ns: u64,
    ordinal: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedWindow {
    #[serde(deserialize_with = "read_residuals")]
    samples: Vec<SavedResidual>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PersistedCalibration<D> {
    pub model_revision: u64,
    pub domain: D,
    global: SavedWindow,
    #[serde(deserialize_with = "read_buckets")]
    buckets: Vec<(u32, SavedWindow)>,
}

fn read_residuals<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<SavedResidual>, D::Error> {
    bounded_sequence::<D, SavedResidual, 4096>(d)
}

fn read_buckets<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Vec<(u32, SavedWindow)>, D::Error> {
    bounded_sequence::<D, (u32, SavedWindow), 9>(d)
}

pub(crate) fn bounded_sequence<'de, D, T, const MAX: usize>(
    deserializer: D,
) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Visitor<T, const MAX: usize>(std::marker::PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const MAX: usize> serde::de::Visitor<'de> for Visitor<T, MAX> {
        type Value = Vec<T>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "at most {MAX} entries")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> Result<Self::Value, A::Error> {
            if sequence.size_hint().is_some_and(|size| size > MAX) {
                return Err(serde::de::Error::custom("collection limit exceeded"));
            }
            // Keep capacities bounded without reserving the full hard ceiling
            // for every small window. Never let geometric growth exceed MAX.
            let mut values = Vec::new();
            while values.len() < MAX {
                let Some(value) = sequence.next_element()? else {
                    return Ok(values);
                };
                if values.len() == values.capacity() {
                    values.reserve_exact((MAX - values.len()).min(16));
                }
                values.push(value);
            }
            if sequence.next_element::<serde::de::IgnoredAny>()?.is_some() {
                return Err(serde::de::Error::custom("collection limit exceeded"));
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Visitor::<T, MAX>(std::marker::PhantomData))
}

impl WindowSnapshot {
    fn persist(
        &self,
        now: Instant,
        config: CalibrationConfig,
    ) -> Result<SavedWindow, &'static str> {
        if self.samples.len() > 4096 {
            return Err("calibration persistence sample limit");
        }
        let mut samples = Vec::with_capacity(self.samples.len());
        for residual in self.fresh(now, config.ttl) {
            let age = now
                .checked_duration_since(residual.observed_at)
                .ok_or("future calibration observation")?;
            samples.push(SavedResidual {
                seconds: residual.seconds,
                ordinal: residual.ordinal,
                age_ns: age
                    .as_nanos()
                    .try_into()
                    .map_err(|_| "calibration age overflow")?,
            });
        }
        Ok(SavedWindow { samples })
    }
}

impl<D: Clone + PartialEq> CalibrationSnapshot<D> {
    pub fn persist(
        &self,
        domain: &D,
        now: Instant,
    ) -> Result<PersistedCalibration<D>, &'static str> {
        let (_, global, buckets) = self
            .domains
            .iter()
            .find(|(key, _, _)| key == domain)
            .ok_or("missing calibration domain")?;
        if buckets.len() > 9 {
            return Err("calibration persistence bucket limit");
        }
        Ok(PersistedCalibration {
            model_revision: self.model_revision,
            domain: domain.clone(),
            global: global.persist(now, self.config)?,
            buckets: buckets
                .iter()
                .map(|(key, window)| Ok((*key, window.persist(now, self.config)?)))
                .collect::<Result<_, &'static str>>()?,
        })
    }
}

impl SavedWindow {
    fn restore(
        self,
        config: CalibrationConfig,
        now: Instant,
        elapsed: Duration,
    ) -> Result<(Window, usize), &'static str> {
        if self.samples.len() > config.window_samples {
            return Err("persisted window exceeds configured bound");
        }
        let mut samples = self.samples;
        samples.sort_unstable_by_key(|sample| sample.ordinal);
        if samples
            .windows(2)
            .any(|pair| pair[0].ordinal == pair[1].ordinal)
        {
            return Err("duplicate residual ordinal");
        }
        let mut window = Window::new(config);
        let mut expired = 0;
        for sample in samples {
            if !sample.seconds.is_finite() || sample.ordinal == 0 || sample.ordinal == u64::MAX {
                return Err("invalid persisted residual");
            }
            let Some(age) = Duration::from_nanos(sample.age_ns).checked_add(elapsed) else {
                expired += 1;
                continue;
            };
            if age >= config.ttl {
                expired += 1;
                continue;
            }
            let observed_at = now
                .checked_sub(age)
                .ok_or("restored monotonic age is unrepresentable")?;
            window.0.push_back(Residual {
                seconds: sample.seconds,
                observed_at,
                ordinal: sample.ordinal,
            });
        }
        Ok((window, expired))
    }
}

impl<D: Clone + PartialEq> PersistedCalibration<D> {
    pub fn restore(
        self,
        config: CalibrationConfig,
        expected_domain: &D,
        revision: u64,
        now: Instant,
        elapsed: Duration,
    ) -> Result<(OnlineCalibrator<D>, usize), &'static str> {
        config.validate()?;
        if self.model_revision != revision
            || revision == 0
            || &self.domain != expected_domain
            || self.buckets.len() > config.buckets_per_domain
        {
            return Err("persisted calibration version/domain/bucket mismatch");
        }
        let (global, mut expired) = self.global.restore(config, now, elapsed)?;
        let mut buckets = Vec::with_capacity(config.buckets_per_domain);
        for (key, saved) in self.buckets {
            if key as usize >= config.buckets_per_domain
                || buckets.iter().any(|(prior, _)| *prior == key)
            {
                return Err("invalid persisted calibration bucket key");
            }
            let (window, missing) = saved.restore(config, now, elapsed)?;
            expired += missing;
            // Repeated ordinals in the global and conditional view must refer
            // to the same observation; a saved bucket cannot rewrite a label.
            for residual in &window.0 {
                if global.0.iter().any(|prior| {
                    prior.ordinal == residual.ordinal
                        && (prior.seconds != residual.seconds
                            || prior.observed_at != residual.observed_at)
                }) {
                    return Err("conflicting persisted residual views");
                }
            }
            buckets.push((key, window));
        }
        let mut ordinals =
            Vec::with_capacity(buckets.iter().map(|(_, window)| window.0.len()).sum());
        for (_, window) in &buckets {
            ordinals.extend(window.0.iter().map(|sample| sample.ordinal));
        }
        ordinals.sort_unstable();
        if ordinals.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("one persisted observation appears in multiple buckets");
        }
        if ordinals
            .last()
            .is_some_and(|last| global.0.back().is_none_or(|global| global.ordinal < *last))
        {
            return Err("persisted global window omits newer bucket observations");
        }
        let next_ordinal = global
            .0
            .iter()
            .chain(buckets.iter().flat_map(|(_, window)| window.0.iter()))
            .map(|sample| sample.ordinal)
            .max()
            .unwrap_or(0);
        let mut calibrator = OnlineCalibrator::new(config, revision)?;
        calibrator.next_ordinal = next_ordinal;
        calibrator.domains.push(DomainWindow {
            domain: self.domain,
            global,
            buckets,
        });
        Ok((calibrator, expired))
    }
}
