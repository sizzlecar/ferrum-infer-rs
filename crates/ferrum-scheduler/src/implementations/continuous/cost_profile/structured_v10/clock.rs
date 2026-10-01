use super::*;

#[cfg(test)]
mod same_boot_tests {
    use super::*;
    use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;

    #[test]
    fn same_boot_age_uses_each_original_child_and_inclusive_declared_limits() {
        let original = CostMonotonicDomainV1::new_linux_boottime(
            [1; 16],
            7,
            std::num::NonZeroU64::new(9).unwrap(),
            0,
            0,
        )
        .unwrap();
        let current = original.clone();
        let limits = CostProfileLoadLimits {
            max_profile_age_ns: std::num::NonZeroU64::new(20).unwrap(),
            ..Default::default()
        };
        let evidence = Evidence {
            opening: PairedClock {
                monotonic_ns: 100,
                wall_unix_ns: 10_000,
            },
            closing: PairedClock {
                monotonic_ns: 180,
                wall_unix_ns: 1,
            },
            oldest_observed: 120,
            newest_observed: 160,
            max_age_ns: 80,
        };
        let mapped = same_boot(evidence, Some(&original), &current, 200, 0, &limits).unwrap();
        assert_eq!((mapped.oldest_age, mapped.newest_age), (80, 40));
        assert_eq!(
            (
                mapped.clock.source_monotonic_anchor_ns,
                mapped.clock.model_anchor_ns
            ),
            (180, 180)
        );
        assert!(same_boot(evidence, Some(&original), &current, 201, 0, &limits).is_err());
        // A fresh sibling cannot refresh the older child or erase its age.
        let older = Evidence {
            oldest_observed: 119,
            ..evidence
        };
        assert!(same_boot(older, Some(&original), &current, 200, 0, &limits).is_err());
        assert!(same_boot(evidence, None, &current, 200, 0, &limits).is_err());
        assert!(same_boot(evidence, Some(&original), &current, 179, 0, &limits).is_err());
        let other_offset = CostMonotonicDomainV1::new_linux_boottime(
            [1; 16],
            7,
            std::num::NonZeroU64::new(9).unwrap(),
            1,
            0,
        )
        .unwrap();
        assert!(same_boot(evidence, Some(&original), &other_offset, 200, 0, &limits).is_err());
    }
}

pub(super) struct Mapped {
    pub clock: ProfileObservationClock,
    pub model_now: u64,
    pub error: u64,
    pub oldest_age: u64,
    pub newest_age: u64,
    pub wall: u64,
}
#[derive(Clone, Copy)]
pub(super) struct Evidence {
    pub opening: PairedClock,
    pub closing: PairedClock,
    pub oldest_observed: u64,
    pub newest_observed: u64,
    pub max_age_ns: u64,
}
impl From<&replay::Replayed> for Evidence {
    fn from(r: &replay::Replayed) -> Self {
        Self {
            opening: r.header.opening,
            closing: r.closing,
            oldest_observed: r.oldest_observed,
            newest_observed: r.newest_observed,
            max_age_ns: r.header.settings.max_age_ns,
        }
    }
}

/// Only for a collector owned by the same live runtime clock. A persisted
/// source cannot establish this relationship; its loader must use map_clock.
pub(super) fn same_process(
    r: Evidence,
    now: u64,
    wall: u64,
    limits: &CostProfileLoadLimits,
) -> Result<Mapped, CostProfileError> {
    let fail = || CostProfileError::Clock("live source clock differs or original evidence expired");
    if r.oldest_observed < r.opening.monotonic_ns
        || r.newest_observed < r.oldest_observed
        || r.closing.monotonic_ns < r.newest_observed
        || now
            .checked_sub(r.closing.monotonic_ns)
            .is_none_or(|age| age > limits.max_profile_age_ns.get())
    {
        return Err(fail());
    }
    let oldest_age = now.checked_sub(r.oldest_observed).ok_or_else(fail)?;
    let newest_age = now.checked_sub(r.newest_observed).ok_or_else(fail)?;
    if oldest_age > r.max_age_ns {
        return Err(fail());
    }
    Ok(Mapped {
        clock: ProfileObservationClock {
            source_monotonic_anchor_ns: now,
            model_anchor_ns: now,
        },
        model_now: now,
        error: 0,
        oldest_age,
        newest_age,
        // Informational receipt timestamp, never used for age or freshness.
        wall,
    })
}
/// Persisted reuse requires the original OS clock identity. It cannot derive
/// that identity from metadata, wall clocks, or a fresh process-local anchor.
pub(super) fn same_boot(
    r: Evidence,
    original: Option<&ferrum_interfaces::execution_cost::CostMonotonicDomainV1>,
    current: &ferrum_interfaces::execution_cost::CostMonotonicDomainV1,
    now: u64,
    wall: u64,
    limits: &CostProfileLoadLimits,
) -> Result<Mapped, CostProfileError> {
    current
        .validate()
        .map_err(|_| CostProfileError::Clock("invalid current monotonic domain"))?;
    if original != Some(current) {
        return Err(CostProfileError::Clock(
            "original boot or clock namespace differs",
        ));
    }
    let fail = || CostProfileError::Clock("boot source clock differs or original evidence expired");
    if r.oldest_observed < r.opening.monotonic_ns
        || r.newest_observed < r.oldest_observed
        || r.closing.monotonic_ns < r.newest_observed
        || now
            .checked_sub(r.closing.monotonic_ns)
            .is_none_or(|age| age > limits.max_profile_age_ns.get())
    {
        return Err(fail());
    }
    let oldest_age = now.checked_sub(r.oldest_observed).ok_or_else(fail)?;
    let newest_age = now.checked_sub(r.newest_observed).ok_or_else(fail)?;
    if oldest_age > r.max_age_ns {
        return Err(fail());
    }
    Ok(Mapped {
        clock: ProfileObservationClock {
            source_monotonic_anchor_ns: r.closing.monotonic_ns,
            model_anchor_ns: r.closing.monotonic_ns,
        },
        // No wall-clock translation or declared wall-clock accuracy is involved.
        model_now: now,
        error: 0,
        oldest_age,
        newest_age,
        wall,
    })
}
pub(super) fn validate_source(
    r: impl Into<Evidence>,
    source_error: u64,
) -> Result<(), CostProfileError> {
    let r = r.into();
    let fail = || CostProfileError::Clock("inconsistent original paired clocks");
    let monotonic = r
        .closing
        .monotonic_ns
        .checked_sub(r.opening.monotonic_ns)
        .ok_or_else(fail)?;
    let wall = r
        .closing
        .wall_unix_ns
        .checked_sub(r.opening.wall_unix_ns)
        .ok_or_else(fail)?;
    if wall
        .checked_add(source_error.checked_mul(2).ok_or_else(fail)?)
        .ok_or_else(fail)?
        < monotonic
    {
        return Err(fail());
    }
    Ok(())
}
pub(super) fn map_clock(
    r: impl Into<Evidence>,
    source_error: u64,
    limits: &CostProfileLoadLimits,
    load: ProfileLoadClock,
) -> Result<Mapped, CostProfileError> {
    let r = r.into();
    let fail =
        || CostProfileError::Clock("invalid structured paired clock or original evidence expired");
    validate_source(r, source_error)?;
    let wall = load.wall_unix_ns.filter(|n| *n > 0).ok_or_else(fail)?;
    let local_error = load.wall_max_error_ns.ok_or_else(fail)?;
    if source_error > limits.max_clock_error_ns || local_error > limits.max_clock_error_ns {
        return Err(fail());
    }
    let error = source_error.checked_add(local_error).ok_or_else(fail)?;
    if error > limits.max_clock_error_ns {
        return Err(fail());
    }
    // Opening wall is read before monotonic. Its offset is a lower bound;
    // using it makes observations conservatively older. Closing bounds the
    // opposite end and checks source-clock consistency, never renews its age.
    let opening = r.opening;
    let close = r.closing;
    let mono_delta = close
        .monotonic_ns
        .checked_sub(opening.monotonic_ns)
        .ok_or_else(fail)?;
    let wall_delta = close
        .wall_unix_ns
        .checked_sub(opening.wall_unix_ns)
        .ok_or_else(fail)?;
    if wall_delta
        .checked_add(source_error.checked_mul(2).ok_or_else(fail)?)
        .ok_or_else(fail)?
        < mono_delta
    {
        return Err(fail());
    }
    let elapsed = wall
        .checked_sub(opening.wall_unix_ns)
        .and_then(|n| n.checked_add(error))
        .ok_or_else(fail)?;
    if wall
        .checked_sub(close.wall_unix_ns)
        .and_then(|n| n.checked_add(error))
        .is_none_or(|n| n > limits.max_profile_age_ns.get())
    {
        return Err(fail());
    }
    let anchor = opening.monotonic_ns.checked_add(elapsed).ok_or_else(fail)?;
    if anchor < close.monotonic_ns {
        return Err(fail());
    }
    let oldest_age = anchor.checked_sub(r.oldest_observed).ok_or_else(fail)?;
    let newest_age = anchor.checked_sub(r.newest_observed).ok_or_else(fail)?;
    if oldest_age > r.max_age_ns {
        return Err(fail());
    }
    Ok(Mapped {
        clock: ProfileObservationClock {
            source_monotonic_anchor_ns: load.monotonic_now_ns,
            model_anchor_ns: anchor,
        },
        model_now: anchor,
        error,
        oldest_age,
        newest_age,
        wall,
    })
}
