//! Activation checks against the current runtime's complete cold descriptor.
//! A source's self-consistent identity does not establish its capacity limits.
use super::*;
use ferrum_interfaces::execution_cost::CostWorkloadDomainV1;

pub(super) fn validate_runtime_domain(
    declared: Option<&CostWorkloadDomainV1>,
    runtime: Option<&CostWorkloadDomainV1>,
) -> Result<(), FerrumError> {
    let Some(declared) = declared else {
        return Ok(());
    };
    let runtime = runtime.ok_or_else(|| {
        FerrumError::config(
            "physical-envelope profile requires a known current runtime workload domain",
        )
    })?;
    if !declared.matches_runtime_domain(runtime) {
        return Err(FerrumError::config(
            "physical-envelope profile workload limits or projection differ from current runtime",
        ));
    }
    Ok(())
}

fn same_declared_domain(
    parent: Option<&CostWorkloadDomainV1>,
    child: Option<&CostWorkloadDomainV1>,
) -> bool {
    match (parent, child) {
        (None, None) => true,
        (Some(parent), Some(child)) => parent.matches_runtime_domain(child),
        _ => false,
    }
}

pub(super) fn validate_service_catalog(
    imported: &file::ImportedStructuredCatalogV13,
) -> Result<(), FerrumError> {
    let declared = imported.workload_domain();
    if imported
        .children
        .iter()
        .any(|child| !same_declared_domain(declared, child.workload_domain()))
    {
        return Err(FerrumError::config(
            "service catalog child workload domain differs from its frozen declaration",
        ));
    }
    Ok(())
}

pub(super) fn validate_owner_block_catalog(
    imported: &file::ImportedStructuredCatalogV14,
) -> Result<(), FerrumError> {
    let declared = imported.workload_domain();
    if imported.children.is_empty() || imported.children.len() > 128 {
        return Err(FerrumError::config("owner block catalog child capacity"));
    }
    for child in &imported.children {
        let p = child.provenance();
        if !same_declared_domain(declared, child.workload_domain())
            || child.fingerprint() != imported.children[0].fingerprint()
            || p.schema_version != 14
            || p.file_sha256 != imported.file_sha256
            || p.file_bytes != imported.file_bytes
            || p.source_sha256 != imported.source_sha256
            || p.source_bytes != imported.source_bytes
            || p.protocol != imported.capture_protocol
            || p.offered_attempts != imported.offered_attempts
            || p.total_shape_rows != imported.total_shape_rows
        {
            return Err(FerrumError::config(
                "owner block catalog child differs from its frozen source or workload domain",
            ));
        }
    }
    Ok(())
}

/// Profile15 is a distinct preparation protocol. A legacy replayed child
/// cannot acquire its authority merely by appearing in the returned vector.
pub(super) fn validate_prepared_owner_block_catalog(
    imported: &file::ImportedPreparedOwnerBlockCatalogV15,
) -> Result<(), FerrumError> {
    if imported.children.is_empty() || imported.children.len() > 128 {
        return Err(FerrumError::config(
            "prepared owner block catalog child capacity",
        ));
    }
    let declared = imported.workload_domain().ok_or_else(|| {
        FerrumError::config("prepared owner block catalog lacks its physical workload domain")
    })?;
    let fingerprint = imported.children[0].fingerprint();
    let identity = ferrum_interfaces::execution_cost::ExecutorCostIdentity {
        schema_version: 1,
        model_weights: fingerprint.model_weights,
        numerical_policy: fingerprint.numerical_policy,
        device_runtime: fingerprint.device_runtime,
        execution_config: fingerprint.execution_config,
    };
    if !declared.matches_execution_identity(&identity) {
        return Err(FerrumError::config(
            "prepared owner block catalog fingerprint differs from its physical workload domain",
        ));
    }
    for child in &imported.children {
        let p = child.provenance();
        if !same_declared_domain(Some(declared), child.workload_domain())
            || child.fingerprint() != fingerprint
            || p.schema_version != 15
            || p.file_sha256 != imported.file_sha256
            || p.file_bytes != imported.file_bytes
            || p.source_sha256 != imported.source_sha256
            || p.source_bytes != imported.source_bytes
            || p.protocol != imported.capture_protocol
            || p.offered_attempts != imported.offered_attempts
            || p.total_shape_rows != imported.total_shape_rows
            || p.capture_identity != imported.children[0].provenance().capture_identity
            || p.cohort_manifest_sha256 != imported.children[0].provenance().cohort_manifest_sha256
        {
            return Err(FerrumError::config(
                "prepared owner block catalog child differs from its original source or workload domain",
            ));
        }
    }
    Ok(())
}

impl EngineCostSnapshot {
    /// Called before a loaded seed or replacement snapshot can be installed.
    /// Legacy children preserve their previous semantics. Every physical child
    /// must independently match the real runtime, including in mixed catalogs.
    pub(in crate::continuous_engine::inner::cost_observation) fn validate_workload_domain(
        &self,
        runtime: Option<&CostWorkloadDomainV1>,
    ) -> Result<(), FerrumError> {
        if let Snapshot::StructuredV2(snapshot) = &self.inner {
            for child in snapshot.children.values() {
                validate_runtime_domain(child.workload_domain(), runtime)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_interfaces::execution_cost::{CostWorkloadLimitsV1, ExecutorCostIdentity};
    use std::num::{NonZeroU32, NonZeroU64};

    fn identity() -> ExecutorCostIdentity {
        ExecutorCostIdentity {
            schema_version: 1,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        }
    }
    fn domain() -> CostWorkloadDomainV1 {
        CostWorkloadDomainV1::new_vnext(
            &identity(),
            CostWorkloadLimitsV1 {
                maximum_rows: NonZeroU32::new(8).unwrap(),
                maximum_context_tokens: NonZeroU32::new(512).unwrap(),
                maximum_scheduled_tokens_per_wave: NonZeroU64::new(128).unwrap(),
                output_vocabulary_elements: NonZeroU64::new(1024).unwrap(),
                repetition_slot_capacity: 512,
                fixed_state_bytes_per_row: 64,
            },
        )
        .unwrap()
    }
    #[test]
    fn activation_requires_the_whole_runtime_domain_not_only_four_identity_digests() {
        let runtime = domain();
        assert!(validate_runtime_domain(Some(&runtime), Some(&runtime)).is_ok());
        assert!(validate_runtime_domain(Some(&runtime), None).is_err());
        for axis in 0..6 {
            let mut limits = runtime.limits().clone();
            match axis {
                0 => limits.maximum_rows = NonZeroU32::new(9).unwrap(),
                1 => limits.maximum_context_tokens = NonZeroU32::new(513).unwrap(),
                2 => limits.maximum_scheduled_tokens_per_wave = NonZeroU64::new(129).unwrap(),
                3 => limits.output_vocabulary_elements = NonZeroU64::new(1025).unwrap(),
                4 => limits.repetition_slot_capacity = 511,
                _ => limits.fixed_state_bytes_per_row = 65,
            }
            let different = CostWorkloadDomainV1::new_vnext(&identity(), limits).unwrap();
            assert!(different.matches_execution_identity(&identity()));
            assert!(validate_runtime_domain(Some(&different), Some(&runtime)).is_err());
            assert!(validate_runtime_domain(Some(&runtime), Some(&different)).is_err());
        }
    }
    #[test]
    fn activation_preserves_legacy_and_binds_each_service_child_to_its_own_header() {
        let runtime = domain();
        assert!(validate_runtime_domain(None, None).is_ok());
        assert!(validate_runtime_domain(None, Some(&runtime)).is_ok());
        assert!(same_declared_domain(None, None));
        assert!(!same_declared_domain(None, Some(&runtime)));
        assert!(!same_declared_domain(Some(&runtime), None));
        assert!(same_declared_domain(Some(&runtime), Some(&runtime)));
    }
}
