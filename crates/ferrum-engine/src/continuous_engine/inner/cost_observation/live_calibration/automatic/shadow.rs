//! A complete old numerical offer population can discover the next catalogue.
//! Only original input facts enter discovery. No timing or fitted parameter is
//! carried across generations, and the next Fit starts after the closing FIFO.
use super::*;

#[derive(Debug, Clone, Serialize)]
pub(super) struct ShadowDiscoveryOrigin {
    pub generation: u64,
    pub phase: usize,
    pub population: tickets::WindowAudit,
    pub opening_fifo_cutoff: u64,
    pub closing_fifo_cutoff: u64,
    pub frozen_at_ns: u64,
}

pub(super) struct ShadowDiscoverySeed {
    pub frozen: discovery::FrozenDiscovery,
    pub origin: ShadowDiscoveryOrigin,
}

impl ShadowDiscoverySeed {
    pub fn current(&self, now: u64, cutoff: u64, maximum_age_ns: u64) -> bool {
        cutoff >= self.origin.closing_fifo_cutoff
            && now
                .checked_sub(self.origin.frozen_at_ns)
                .is_some_and(|age| age <= maximum_age_ns)
    }
}

/// One bounded input-only window, owned by the same worker as the numerical
/// collector. Unequal configured quotas retain the original discovery path;
/// a shorter phase never stands in for the declared discovery population.
pub(super) struct ShadowDiscovery {
    window: Option<discovery::DiscoveryWindow>,
}

impl ShadowDiscovery {
    pub fn open(policy: discovery::DiscoveryPolicy, offers: usize, cutoff: u64) -> Self {
        Self {
            window: (policy.offered_waves == offers)
                .then(|| discovery::DiscoveryWindow::new(policy, cutoff).ok())
                .flatten(),
        }
    }

    pub fn observe(&mut self, ticket: u64, fifo: u64, input: Option<&StructuredInputV2>) {
        if let Some(window) = &mut self.window {
            // DiscoveryWindow permanently remembers its first rejection. A
            // shadow capacity/error must not invalidate the original training
            // population or turn a partial shadow into a reusable seed.
            let _ = match input {
                Some(input) => window.observe(ticket, fifo, input),
                None => window.observe_outside(ticket, fifo),
            };
        }
    }

    pub fn freeze(
        self,
        population: tickets::WindowAudit,
        now: u64,
        cutoff: u64,
    ) -> Option<ShadowDiscoverySeed> {
        if population.failed
            || !population.closed
            || population.issued != population.declared_offers
            || population.retired != population.issued
        {
            return None;
        }
        let window = self.window?;
        if window.offered() != population.issued {
            return None;
        }
        let frozen = window.freeze().ok()?;
        let (opening_fifo_cutoff, closing_fifo_cutoff) = frozen.fifo_bounds();
        if frozen.scopes().is_empty() || cutoff < closing_fifo_cutoff {
            return None;
        }
        Some(ShadowDiscoverySeed {
            frozen,
            origin: ShadowDiscoveryOrigin {
                generation: population.generation,
                phase: population.phase,
                population,
                opening_fifo_cutoff,
                closing_fifo_cutoff,
                frozen_at_ns: now,
            },
        })
    }
}
