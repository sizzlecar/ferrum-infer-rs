//! Peaks for sequential visual and causal owners. PNG ingress and output
//! credits stay with their existing parent charges, outside this phase grant.
use super::*;
use crate::image_generation::p_h3::expansion::vision::ExpansionVisionHostRequirements;

pub(super) struct InitializationRequirements {
    budget: ExecutionHostBudget,
    pub startup_device_bytes: u64,
    // Both original warm-backing assumptions survive through initialization.
    _causal: ExpansionHostRequirements,
    _visual: Option<ExpansionVisionHostRequirements>,
}
impl InitializationRequirements {
    pub fn new(
        causal: ExpansionHostRequirements,
        visual: Option<ExpansionVisionHostRequirements>,
        decoded_rgba: usize,
    ) -> Result<Self> {
        let budget = match &visual {
            Some(visual) => sequential_budget(causal.budget, visual.budget, decoded_rgba)?,
            None => causal.budget,
        };
        let startup_device_bytes = causal
            .startup_device_bytes
            .max(visual.as_ref().map_or(0, |v| v.startup_device_bytes));
        Ok(Self {
            budget,
            startup_device_bytes,
            _causal: causal,
            _visual: visual,
        })
    }
    pub fn claims(&self) -> [(HostMemoryClass, usize); 4] {
        self.budget.claims()
    }
}

fn sequential_budget(
    causal: ExecutionHostBudget,
    visual: ExecutionHostBudget,
    decoded_rgba: usize,
) -> Result<ExecutionHostBudget> {
    // RGBA dies after visual forward. Its F32 output and the causal persistent
    // F16 inputs overlap during the bridge, in distinct Host classes. Causal
    // Input already includes that entire persistent bridge plus wave uploads.
    Ok(ExecutionHostBudget {
        input: causal.input.max(
            visual
                .input
                .checked_add(decoded_rgba)
                .context("visual PNG decode and preprocessing capacity overflow")?,
        ),
        output: causal.output.max(visual.output),
        // Original visual close acknowledgement precedes causal installation.
        staging: causal.staging.max(visual.staging),
        // Materializers may retain converted backing across phases. Preserve
        // both real requirements rather than assuming a cache was released.
        weights: causal
            .weights
            .checked_add(visual.weights)
            .context("visual and causal WeightBacking capacity overflow")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admitted_phase_keeps_ingress_and_output_credit_outside_sequential_work() {
        let geometry = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                4096,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 4096,
                    output_bytes: 4096,
                    transfer_staging_bytes: 4096,
                    weight_backing_bytes: 4096,
                },
            )
            .unwrap(),
            16,
        )
        .unwrap();
        // Query actual allocator backing, including both alignment padding
        // and the leading-offset allowance; rounded payload alone is smaller.
        let physical = |bytes| geometry.backing_bytes(bytes).unwrap();
        let causal = ExecutionHostBudget {
            input: physical(17) + physical(57),
            output: physical(21),
            staging: physical(16),
            weights: physical(19),
        };
        let visual = ExecutionHostBudget {
            input: physical(17),
            output: physical(33),
            staging: physical(32),
            weights: physical(5),
        };
        let caps = HostMemoryClassLimits {
            input_preprocess_bytes: physical(3) + physical(17) + physical(57),
            output_bytes: physical(1) + physical(33),
            transfer_staging_bytes: physical(32),
            weight_backing_bytes: physical(19) + physical(5),
        };
        let total = caps.input_preprocess_bytes
            + caps.output_bytes
            + caps.transfer_staging_bytes
            + caps.weight_backing_bytes;
        let retained_ingress = physical(3) + physical(1);
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(total, caps).unwrap(),
            16,
        )
        .unwrap();
        let ingress = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 3)
            .unwrap();
        let output_credit = host
            .reserve_scope(&[(HostMemoryClass::Output, physical(1))])
            .unwrap();
        let budget = sequential_budget(causal, visual, host.backing_bytes(33).unwrap()).unwrap();
        let phase = host.reserve_scope(&budget.claims()).unwrap();
        let work = phase.account();
        let rgba = work
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 33)
            .unwrap();
        let patches = work
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 17)
            .unwrap();
        let output = work.allocate_zeroed(HostMemoryClass::Output, 33).unwrap();
        drop((rgba, patches));
        let persistent = work
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 17)
            .unwrap();
        // The real F32 output and separately charged causal bridge coexist.
        assert_eq!(host.usage().unwrap().committed_bytes(), total);
        drop(output);
        let wave = work
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 57)
            .unwrap();
        assert!(work
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
            .is_err());
        drop((persistent, wave, work, phase));
        assert_eq!(host.usage().unwrap().committed_bytes(), retained_ingress);
        drop((ingress, output_credit));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }

    #[test]
    fn overflowing_decode_or_weight_requirements_are_rejected() {
        let causal = ExecutionHostBudget {
            input: 140,
            output: 30,
            staging: 7,
            weights: 90,
        };
        let visual = ExecutionHostBudget {
            input: 100,
            output: 110,
            staging: 9,
            weights: 60,
        };
        assert!(sequential_budget(causal, visual, usize::MAX).is_err());
        assert!(sequential_budget(
            ExecutionHostBudget {
                weights: usize::MAX,
                ..causal
            },
            visual,
            0,
        )
        .is_err());
    }
}
