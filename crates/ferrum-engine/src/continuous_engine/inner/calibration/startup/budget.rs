//! Probe admission estimates reserve a complete selected protocol. They do
//! not enter reference measurements or establish a latency guarantee.
use super::*;
use tokio::time::Instant;

pub(super) struct ProbeBudget {
    maximum: usize,
    used: usize,
    reserved: usize,
    time_reservation: Option<(Instant, Duration)>,
}

impl ProbeBudget {
    pub fn new(maximum: NonZeroUsize) -> Self {
        Self {
            maximum: maximum.get(),
            used: 0,
            reserved: 0,
            time_reservation: None,
        }
    }

    pub fn can_fit(&self, preparations: usize, reserved: usize) -> bool {
        self.used
            .checked_add(preparations)
            .and_then(|n| n.checked_add(reserved))
            .is_some_and(|n| n <= self.maximum)
    }

    pub fn reserve(&mut self, reserved: usize) -> Result<()> {
        if !self.can_fit(0, reserved) {
            return Err(invalid(
                "automatic reference cannot reserve its complete frozen trials",
            ));
        }
        self.reserved = reserved;
        Ok(())
    }

    pub fn reserve_time(&mut self, deadline: Instant, required: Duration) {
        self.time_reservation = Some((deadline, required));
    }

    pub fn release_reservations(&mut self) {
        self.reserved = 0;
        self.time_reservation = None;
    }

    pub fn claim(&mut self) -> Result<()> {
        if !self.can_fit(1, self.reserved) {
            return Err(invalid(
                "automatic reference exhausted its unreserved probe request budget",
            ));
        }
        if self.time_reservation.is_some_and(|(deadline, required)| {
            Instant::now()
                .checked_add(required)
                .is_none_or(|end| end > deadline)
        }) {
            return Err(invalid(
                "automatic reference retry would consume reserved formal-trial time",
            ));
        }
        self.used += 1;
        Ok(())
    }

    pub fn used(&self) -> usize {
        self.used
    }
    pub fn maximum(&self) -> usize {
        self.maximum
    }
}

pub(super) struct ProbeWorkPlanner {
    total_ns: u128,
    finalization_ns: u128,
    margin_numerator: u128,
    repetitions: usize,
    prefill_sum_ns: u128,
    last_prefill: Option<(u32, u128)>,
    decode_ns: Option<u128>,
}

fn overflow() -> FerrumError {
    invalid("automatic reference work estimate overflow")
}

fn ceil_ratio(value: u128, numerator: u128, denominator: u128) -> Result<u128> {
    let product = value.checked_mul(numerator).ok_or_else(overflow)?;
    let quotient = product / denominator;
    quotient
        .checked_add(u128::from(product % denominator != 0))
        .ok_or_else(overflow)
}

fn duration(ns: u128) -> Result<Duration> {
    Ok(Duration::new(
        u64::try_from(ns / 1_000_000_000).map_err(|_| overflow())?,
        (ns % 1_000_000_000) as u32,
    ))
}

impl ProbeWorkPlanner {
    pub fn new(settings: &SloAutomaticReferenceProbeSettingsV1) -> Result<Self> {
        settings.validate().map_err(invalid)?;
        let total_ns = u128::from(settings.maximum_duration_ms.get())
            .checked_mul(1_000_000)
            .ok_or_else(overflow)?;
        Ok(Self {
            total_ns,
            finalization_ns: ceil_ratio(
                total_ns,
                u128::from(settings.finalization_reserve_percent),
                100,
            )?,
            margin_numerator: 100 + u128::from(settings.work_estimate_margin_percent),
            repetitions: settings.fresh_trials_per_anchor.get(),
            prefill_sum_ns: 0,
            last_prefill: None,
            decode_ns: None,
        })
    }

    pub fn record_prefill(&mut self, tokens: u32, measured: Duration) -> Result<()> {
        if tokens == 0
            || self
                .last_prefill
                .is_some_and(|(previous, _)| tokens <= previous)
        {
            return Err(invalid(
                "automatic reference budget anchors must be a strict prefix",
            ));
        }
        let cost = measured.as_nanos();
        self.prefill_sum_ns = self.prefill_sum_ns.checked_add(cost).ok_or_else(overflow)?;
        self.last_prefill = Some((tokens, cost));
        Ok(())
    }

    pub fn record_decode(&mut self, measured: Duration) -> Result<()> {
        if self.decode_ns.replace(measured.as_nanos()).is_some() {
            return Err(invalid(
                "automatic reference decode work was already measured",
            ));
        }
        Ok(())
    }

    fn next_ns(&self, tokens: u32) -> Result<u128> {
        let (previous, cost) = self
            .last_prefill
            .ok_or_else(|| invalid("missing reference budget seed"))?;
        if tokens <= previous {
            return Err(invalid(
                "automatic reference next anchor must extend the selected prefix",
            ));
        }
        // Scale the most recent complete probe, not max(ms/token) over tiny
        // anchors whose fixed decode/host overhead would dominate forever.
        ceil_ratio(cost, u128::from(tokens), u128::from(previous))
    }

    fn with_margin(&self, ns: u128) -> Result<Duration> {
        duration(
            ceil_ratio(ns, self.margin_numerator, 100)?
                .checked_add(self.finalization_ns)
                .ok_or_else(overflow)?,
        )
    }

    pub fn complete_required(&self) -> Result<Duration> {
        let unit = self
            .prefill_sum_ns
            .checked_add(
                self.decode_ns
                    .ok_or_else(|| invalid("missing reference decode budget seed"))?,
            )
            .ok_or_else(overflow)?;
        self.with_margin(
            unit.checked_mul(self.repetitions as u128)
                .ok_or_else(overflow)?,
        )
    }

    pub fn next_required(
        &self,
        tokens: u32,
        preparation_requests: usize,
        measured_warmup: Option<Duration>,
    ) -> Result<Duration> {
        let next = self
            .next_ns(tokens)?
            .max(measured_warmup.map_or(0, |n| n.as_nanos()));
        let selected = self
            .prefill_sum_ns
            .checked_add(
                self.decode_ns
                    .ok_or_else(|| invalid("missing reference decode budget seed"))?,
            )
            .and_then(|n| n.checked_add(next))
            .ok_or_else(overflow)?;
        let formal = selected
            .checked_mul(self.repetitions as u128)
            .ok_or_else(overflow)?;
        let preparation = next
            .checked_mul(preparation_requests as u128)
            .ok_or_else(overflow)?;
        self.with_margin(formal.checked_add(preparation).ok_or_else(overflow)?)
    }

    pub fn fits(&self, elapsed: Duration, required: Duration) -> bool {
        elapsed
            .as_nanos()
            .checked_add(required.as_nanos())
            .is_some_and(|n| n <= self.total_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planner() -> ProbeWorkPlanner {
        ProbeWorkPlanner::new(&SloAutomaticReferenceProbeSettingsV1::default()).unwrap()
    }

    #[test]
    fn work_budget_stops_expansion_with_time_remaining_and_reserves_all_three_trials() {
        let mut p = planner();
        p.record_prefill(1, Duration::from_millis(300)).unwrap();
        p.record_prefill(1024, Duration::from_secs(5)).unwrap();
        p.record_decode(Duration::from_millis(200)).unwrap();
        let complete = p.complete_required().unwrap();
        assert_eq!(complete, Duration::from_millis(32_625));
        let next = p.next_required(2048, 2, None).unwrap();
        assert_eq!(next, Duration::from_millis(95_125));
        assert!(p.fits(Duration::from_secs(30), complete));
        assert!(!p.fits(Duration::from_secs(30), next));
        // A tiny fixed-overhead anchor is not the perpetual per-token rate.
        assert!(p.fits(Duration::ZERO, next));
    }

    #[test]
    fn work_estimates_round_up_and_never_wrap_or_admit_equal_anchor() {
        assert_eq!(ceil_ratio(1, 125, 100).unwrap(), 2);
        assert!(ceil_ratio(u128::MAX, 125, 100).is_err());
        let mut p = planner();
        p.record_prefill(2, Duration::from_nanos(1)).unwrap();
        p.record_decode(Duration::ZERO).unwrap();
        assert_eq!(p.next_ns(3).unwrap(), 2);
        assert!(p.next_required(2, 2, None).is_err());
        assert!(!p.fits(Duration::MAX, Duration::from_nanos(1)));
    }

    #[test]
    fn fresh_discovery_retries_cannot_spend_reserved_formal_requests() {
        let mut budget = ProbeBudget::new(NonZeroUsize::new(10).unwrap());
        budget.reserve(6).unwrap();
        for _ in 0..4 {
            budget.claim().unwrap();
        }
        assert!(budget.claim().is_err());
        assert_eq!(budget.used(), 4);
        budget.release_reservations();
        for _ in 0..6 {
            budget.claim().unwrap();
        }
        assert!(budget.claim().is_err());
    }

    #[tokio::test(start_paused = true)]
    async fn discovery_retry_uses_original_clock_and_cannot_consume_formal_time() {
        let mut budget = ProbeBudget::new(NonZeroUsize::new(100).unwrap());
        budget.reserve(6).unwrap();
        budget.reserve_time(
            Instant::now() + Duration::from_secs(20),
            Duration::from_secs(12),
        );
        budget.claim().unwrap();
        tokio::time::advance(Duration::from_secs(9)).await;
        assert!(budget.claim().is_err());
        assert_eq!(
            budget.used(),
            1,
            "rejected retries never allocate another owner"
        );
    }

    #[test]
    fn underestimated_admitted_work_never_removes_an_anchor_to_make_formal_fit() {
        let mut p = planner();
        p.record_prefill(1, Duration::from_millis(100)).unwrap();
        p.record_decode(Duration::from_millis(100)).unwrap();
        assert!(p.fits(Duration::from_secs(1), p.next_required(2, 2, None).unwrap()));
        p.record_prefill(2, Duration::from_secs(25)).unwrap();
        assert!(!p.fits(Duration::from_secs(30), p.complete_required().unwrap()));
        assert_eq!(
            p.last_prefill.unwrap().0,
            2,
            "selected anchors cannot be pruned after observing their costs"
        );
    }
}
