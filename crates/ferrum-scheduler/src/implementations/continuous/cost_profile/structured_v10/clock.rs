use super::*;
pub(super) struct Mapped {
    pub clock: ProfileObservationClock,
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
        error,
        oldest_age,
        newest_age,
        wall,
    })
}
