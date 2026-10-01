//! Preallocation accounting for the fixed, zero-row ExactV1 maintenance domain.
//! This does not admit a new shape or relax any trainer retention limit.
use super::*;
use std::mem::size_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExactMaintenanceMemoryRequirements {
    /// Trainer, all retained buckets/samples and persistent floors, excluding
    /// its separately leased published Arc.
    pub retained_bytes: usize,
    pub snapshot_bytes: usize,
    /// Fresh snapshot plus cloned floors and quantile construction scratch.
    pub publication_peak_bytes: usize,
}

// std's BTree nodes have 11 entries and 12 edges (B=6). Charge an entire
// internal node for EVERY entry plus one root, including padding. This is an
// upper bound for both occupied and partially filled nodes, not an occupancy
// estimate. VecDeque storage below includes simultaneous old/new growth.
fn tree_node_bytes<K, V>() -> Option<usize> {
    size_of::<K>()
        .checked_add(size_of::<V>())?
        .checked_mul(11)?
        .checked_add(16 * size_of::<usize>())
}

pub(super) fn zero_row_maintenance(shape: &WaveExecutionShape) -> bool {
    matches!(shape.kind, WaveKind::Maintenance | WaveKind::Restore)
        && shape.decode_kv_tokens.capacity() == 0
        && shape.prefill_chunks.capacity() == 0
        && shape.numeric_features.is_none()
        && shape.host_content_features.is_none()
        && shape.row_multiset_features.is_none()
}

impl CostModelTrainer {
    /// Called before observe/publish allocations. The caller must retain the
    /// trainer reservation and a separate lease for every published snapshot,
    /// including the Arc retained by this trainer. Only ExactV1 maintenance
    /// shapes without any dynamically allocated feature payload are accepted.
    pub fn exact_maintenance_memory_requirements(
        &self,
        shape: &WaveExecutionShape,
        observed_at_ns: u64,
    ) -> Result<ExactMaintenanceMemoryRequirements, CostModelError> {
        let overflow = CostModelError::ArithmeticOverflow;
        if !matches!(self.settings.feature_model, CostFeatureModel::ExactV1 {})
            || !zero_row_maintenance(shape)
            || self.buckets.keys().any(|k| !zero_row_maintenance(&k.shape))
            || self
                .buckets
                .values()
                .flat_map(|bucket| &bucket.samples)
                .any(|sample| !zero_row_maintenance(&sample.shape))
            || self
                .planning_floors
                .keys()
                .any(|k| !zero_row_maintenance(&k.shape))
        {
            return Err(CostModelError::InvalidShape(
                "memory plan requires zero-row ExactV1 maintenance",
            ));
        }
        if observed_at_ns < self.last_clock_ns {
            return Err(CostModelError::ClockMovedBackwards);
        }
        let canonical = canonical_shape(shape, &self.settings.shape_limits)?;
        let key = bucket_key(
            &canonical,
            CostBoundary::PreparationToCommit,
            &self.settings,
        );
        let expiry = self.expiry_plan(&key, observed_at_ns)?;
        let after = self
            .buckets
            .len()
            .checked_sub(expiry.buckets)
            .and_then(|n| n.checked_add(usize::from(expiry.target_samples == 0)))
            .ok_or(overflow)?;
        if after > self.settings.max_buckets.get() {
            return Err(CostModelError::CapacityExceeded("buckets"));
        }
        let buckets = self.buckets.len().max(after);
        // Empty nodes may survive removal; reserve one complete root as well.
        let nodes = buckets.checked_add(1).ok_or(overflow)?;
        let samples = self
            .settings
            .max_samples_per_bucket
            .get()
            .checked_next_power_of_two()
            .ok_or(overflow)?
            .max(4);
        let training_nodes = tree_node_bytes::<BucketKey, TrainingBucket>().ok_or(overflow)?;
        let floor_nodes = tree_node_bytes::<BucketKey, u64>().ok_or(overflow)?;
        let snapshot_nodes = tree_node_bytes::<BucketKey, CalibratedBucket>().ok_or(overflow)?;
        let samples_bytes = samples
            .checked_mul(2)
            .and_then(|n| n.checked_mul(size_of::<CostSample>()))
            .ok_or(overflow)?;
        let retained_bytes = nodes
            .checked_mul(training_nodes.checked_add(floor_nodes).ok_or(overflow)?)
            .and_then(|n| n.checked_add(buckets.checked_mul(samples_bytes)?))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or(overflow)?;
        let coverage_bytes = size_of::<ObservedShapeCoverage>() + 2 * size_of::<usize>();
        let snapshot_bytes = nodes
            .checked_mul(snapshot_nodes)
            .and_then(|n| n.checked_add(buckets.checked_mul(coverage_bytes)?))
            .and_then(|n| n.checked_add(size_of::<CostModelSnapshot>() + 2 * size_of::<usize>()))
            .ok_or(overflow)?;
        let scratch = samples
            .checked_mul(2)
            .and_then(|n| n.checked_mul(size_of::<&CostSample>() + size_of::<u64>()))
            .ok_or(overflow)?;
        let publication_peak_bytes = snapshot_bytes
            .checked_add(nodes.checked_mul(floor_nodes).ok_or(overflow)?)
            .and_then(|n| n.checked_add(scratch))
            .ok_or(overflow)?;
        Ok(ExactMaintenanceMemoryRequirements {
            retained_bytes,
            snapshot_bytes,
            publication_peak_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shape() -> WaveExecutionShape {
        WaveExecutionShape {
            row_multiset_features: None,
            host_content_features: None,
            numeric_features: None,
            kind: WaveKind::Maintenance,
            path: WaveExecutionPath::PlanRuntime,
            provider_signature: [1; 32],
            output_policy_signature: [2; 32],
            graph_state: WaveGraphState::Disabled,
            order: BatchOrderSemantics::Ordered,
            decode_kv_tokens: Vec::new(),
            prefill_chunks: Vec::new(),
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 64,
            maintenance_units: 1,
        }
    }
    fn trainer() -> CostModelTrainer {
        CostModelTrainer::new(
            ExecutionFingerprint {
                model_weights: [1; 32],
                numerical_policy: [2; 32],
                device_runtime: [3; 32],
                execution_config: [4; 32],
            },
            CostModelSettings {
                feature_model: CostFeatureModel::ExactV1 {},
                ..Default::default()
            },
        )
        .unwrap()
    }
    #[test]
    fn exact_maintenance_memory_rejects_inference_and_dynamic_spare_storage() {
        let trainer = trainer();
        let mut value = shape();
        assert!(trainer
            .exact_maintenance_memory_requirements(&value, 0)
            .is_ok());
        value.kind = WaveKind::Prefill;
        assert!(trainer
            .exact_maintenance_memory_requirements(&value, 0)
            .is_err());
        value = shape();
        value.prefill_chunks = Vec::with_capacity(1);
        assert!(trainer
            .exact_maintenance_memory_requirements(&value, 0)
            .is_err());
        value = shape();
        value.kind = WaveKind::Restore;
        value.maintenance_bytes = 0;
        value.maintenance_units = 0;
        assert!(trainer
            .exact_maintenance_memory_requirements(&value, 0)
            .is_err());
        value.restore_bytes = 64;
        assert!(trainer
            .exact_maintenance_memory_requirements(&value, 0)
            .is_ok());
    }
    #[test]
    fn exact_maintenance_memory_accounts_only_actual_domains_and_publication_peak() {
        let mut trainer = trainer();
        let value = shape();
        let first = trainer
            .exact_maintenance_memory_requirements(&value, 1)
            .unwrap();
        assert!(first.publication_peak_bytes > first.snapshot_bytes);
        let sample = WaveCostObservation {
            fingerprint: trainer.fingerprint.clone(),
            actual_shape: value.clone(),
            boundary: CostBoundary::PreparationToCommit,
            outcome: WaveObservationOutcome::Completed,
            timing: WaveTiming {
                wall_total_ns: 10,
                device_elapsed_ns: None,
                stages: Default::default(),
            },
            observed_at_ns: 1,
        };
        trainer.observe(sample).unwrap();
        assert_eq!(
            first,
            trainer
                .exact_maintenance_memory_requirements(&value, 2)
                .unwrap()
        );
        let mut second = value;
        second.provider_signature = [7; 32];
        let grown = trainer
            .exact_maintenance_memory_requirements(&second, 2)
            .unwrap();
        assert!(grown.retained_bytes > first.retained_bytes);
        assert!(grown.snapshot_bytes > first.snapshot_bytes);
    }
}
