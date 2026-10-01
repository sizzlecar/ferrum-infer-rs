//! In-memory adapters for already qualified children. No wire decoder
//! or public receipt can construct an ImportedStructuredModelV2 here.
use super::super::super::structured_epoch;
use super::*;

pub(super) fn owner_block_snapshot(
    imported: &file::ImportedStructuredCatalogV14,
) -> Result<StructuredSnapshot, FerrumError> {
    domain::validate_owner_block_catalog(imported)?;
    let mut children = BTreeMap::new();
    for child in &imported.children {
        if children
            .values()
            .any(|old: &ImportedStructuredModelV2| old.same_population(child))
            || children
                .insert(*child.domain_signature(), child.clone())
                .is_some()
        {
            return Err(FerrumError::config(
                "duplicate owner block replayed owner or domain",
            ));
        }
    }
    Ok(StructuredSnapshot {
        children: Arc::new(children),
        feedback: None,
        epoch: structured_epoch::View::initial(),
    })
}

pub(super) fn prepared_owner_block_snapshot(
    imported: &file::ImportedPreparedOwnerBlockCatalogV15,
) -> Result<StructuredSnapshot, FerrumError> {
    domain::validate_prepared_owner_block_catalog(imported)?;
    let mut children = BTreeMap::new();
    for child in &imported.children {
        if children
            .values()
            .any(|old: &ImportedStructuredModelV2| old.same_population(child))
            || children
                .insert(*child.domain_signature(), child.clone())
                .is_some()
        {
            return Err(FerrumError::config(
                "duplicate prepared owner block replayed owner or domain",
            ));
        }
    }
    Ok(StructuredSnapshot {
        children: Arc::new(children),
        feedback: None,
        epoch: structured_epoch::View::initial(),
    })
}

impl EngineCostSnapshot {
    /// Time-driven removal is distinct from a new calibration. Revocation
    /// cannot be repaired by selecting a subset of the revoked catalog.
    pub(in crate::continuous_engine::inner::cost_observation) fn expired_live_subset(
        &self,
        now: u64,
    ) -> Result<Option<(usize, Vec<ImportedStructuredModelV2>)>, FerrumError> {
        if !self.current() {
            return Ok(None);
        }
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Ok(None);
        };
        if value
            .children
            .values()
            .all(|child| child.is_current_local(now).is_ok())
        {
            return Ok(None);
        }
        let fresh = self.live_children(now)?;
        Ok((fresh.len() != value.children.len()).then_some((value.children.len(), fresh)))
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn validate_retained_subset(
        &self,
        next: &Self,
    ) -> Result<(), FerrumError> {
        let (Snapshot::StructuredV2(old), Snapshot::StructuredV2(new)) = (&self.inner, &next.inner)
        else {
            return Err(FerrumError::config(
                "retained subset requires structured catalogs",
            ));
        };
        if !self.current()
            || self.fingerprint != next.fingerprint
            || new.children.is_empty()
            || new.children.len() >= old.children.len()
        {
            return Err(FerrumError::config(
                "retained subset is not a strict current subset",
            ));
        }
        for (domain, child) in new.children.iter() {
            let original = old
                .children
                .get(domain)
                .ok_or_else(|| FerrumError::config("retained subset added a new child"))?;
            if receipt::child(original)? != receipt::child(child)? {
                return Err(FerrumError::config(
                    "retained subset changed original provenance",
                ));
            }
        }
        Ok(())
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn retain_startup_catalog_receipt(
        receipt: &mut SloCostProfileReceipt,
        original: &[ImportedStructuredModelV2],
        keep: &[bool],
    ) -> Result<(), FerrumError> {
        receipt::retain_startup_catalog_children(receipt, original, keep)
    }

    #[cfg(test)]
    pub(in crate::continuous_engine::inner) fn check_startup_receipt_subset_for_test(
        &self,
        receipt: &mut SloCostProfileReceipt,
        original: &[ImportedStructuredModelV2],
        keep: &[bool],
    ) -> Result<(), FerrumError> {
        Self::retain_startup_catalog_receipt(receipt, original, keep)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn live_owner_block_catalog_with_domain(
        path: Option<&Path>,
        imported: file::ImportedStructuredCatalogV14,
        workload_domain: Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    ) -> Result<(Vec<ImportedStructuredModelV2>, SloCostProfileReceipt), FerrumError> {
        let snapshot = owner_block_snapshot(&imported)?;
        for child in &imported.children {
            domain::validate_runtime_domain(child.workload_domain(), workload_domain)?;
        }
        let receipt = receipt::owner_blocks(path, &imported, &snapshot, None)?;
        Ok((imported.children, receipt))
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn live_prepared_owner_block_catalog_with_domain(
        path: Option<&Path>,
        imported: file::ImportedPreparedOwnerBlockCatalogV15,
        workload_domain: Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    ) -> Result<(Vec<ImportedStructuredModelV2>, SloCostProfileReceipt), FerrumError> {
        let snapshot = prepared_owner_block_snapshot(&imported)?;
        for child in &imported.children {
            domain::validate_runtime_domain(child.workload_domain(), workload_domain)?;
        }
        let receipt = receipt::prepared_owner_blocks(path, &imported, &snapshot, None)?;
        Ok((imported.children, receipt))
    }
    /// A new owner is outside this immutable catalog. A different domain for
    /// an existing owner remains a coverage failure, never an exclusion.
    pub(in crate::continuous_engine::inner::cost_observation) fn undeclared_structured_owner(
        &self,
        query: &StructuredQueryV2,
    ) -> bool {
        self.current()
            && matches!(&self.inner, Snapshot::StructuredV2(value)
            if value.children.values().all(|child| {
                // A different population is an exclusion only inside the
                // original physical domain. Family keys also contain D, so
                // comparing unequal keys alone would hide a changed domain.
                // Legacy children without D keep their original semantics.
                child.workload_domain().is_none_or(|domain|
                    query.input().physical_domain_signature() == Some(domain.sha256()))
            }) && value.children.values().all(|child| {
                if let Some(family) = child.numerical_family_key() {
                    let key = match child.algorithm_universe() {
                        Some(universe) => query.input().numerical_family_key_for_universe(universe),
                        None => query.input().numerical_family_key(),
                    };
                    return match key {
                        Ok(key) => &key != family,
                        Err(Unknown::UnsupportedScope) => true,
                        Err(Unknown::WrongDomain) => child.algorithm_universe().is_some_and(
                            |universe| matches!(universe.contains_checked_algorithms(query.input()), Ok(false))
                        ),
                        Err(_) => false,
                    };
                }
                query.input().cost_template_identity(child.owner().cost_template_policy())
                    .is_some_and(|(owner, _)| child.owner() != owner)
            }))
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn live_service_catalog(
        path: Option<&Path>,
        imported: file::ImportedStructuredCatalogV13,
    ) -> Result<(Vec<ImportedStructuredModelV2>, SloCostProfileReceipt), FerrumError> {
        Self::live_service_catalog_with_domain(path, imported, None)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn live_service_catalog_with_domain(
        path: Option<&Path>,
        imported: file::ImportedStructuredCatalogV13,
        workload_domain: Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    ) -> Result<(Vec<ImportedStructuredModelV2>, SloCostProfileReceipt), FerrumError> {
        domain::validate_service_catalog(&imported)?;
        for child in &imported.children {
            domain::validate_runtime_domain(child.workload_domain(), workload_domain)?;
        }
        let snapshot = StructuredSnapshot {
            children: Arc::new(
                imported
                    .children
                    .iter()
                    .map(|child| (*child.domain_signature(), child.clone()))
                    .collect(),
            ),
            feedback: None,
            epoch: structured_epoch::View::initial(),
        };
        let receipt = receipt::service(path, &imported, &snapshot, None)?;
        Ok((imported.children, receipt))
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn with_structured_epoch(
        &self,
        epoch: structured_epoch::View,
    ) -> Option<Arc<Self>> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return None;
        };
        let mut value = value.clone();
        value.epoch = epoch;
        Some(Arc::new(Self {
            inner: Snapshot::StructuredV2(value),
            fingerprint: self.fingerprint.clone(),
        }))
    }
    /// Worker-only ownership inventory, including children whose original TTL
    /// may have just elapsed. This grants no prediction/publication authority:
    /// their memory/source slots remain held until the expiry transaction.
    pub(in crate::continuous_engine::inner::cost_observation) fn retained_catalog_children(
        &self,
    ) -> Result<Vec<ImportedStructuredModelV2>, FerrumError> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config(
                "catalog ownership requires a structured V2 runtime",
            ));
        };
        Ok(value.children.values().cloned().collect())
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn live_children(
        &self,
        now: u64,
    ) -> Result<Vec<ImportedStructuredModelV2>, FerrumError> {
        if !self.current() {
            return Ok(Vec::new());
        }
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config(
                "live publication requires a structured V2 runtime",
            ));
        };
        let mut result = Vec::new();
        for child in value.children.values() {
            match child.is_current_local(now) {
                Ok(()) => result.push(child.clone()),
                Err(Unknown::Stale) => {}
                Err(error) => {
                    return Err(profile_error(format!("retained catalog clock: {error:?}")));
                }
            }
        }
        Ok(result)
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn live_catalog(
        children: Vec<ImportedStructuredModelV2>,
        fingerprint: model::ExecutionFingerprint,
        epoch: structured_epoch::View,
        now: u64,
    ) -> Result<Arc<Self>, FerrumError> {
        Self::live_catalog_with_domain(children, fingerprint, epoch, now, None)
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn live_catalog_with_domain(
        children: Vec<ImportedStructuredModelV2>,
        fingerprint: model::ExecutionFingerprint,
        epoch: structured_epoch::View,
        now: u64,
        workload_domain: Option<&ferrum_interfaces::execution_cost::CostWorkloadDomainV1>,
    ) -> Result<Arc<Self>, FerrumError> {
        if children.is_empty() || children.len() > 128 {
            return Err(FerrumError::config("live catalog child capacity"));
        }
        let mut catalog = BTreeMap::new();
        for child in children {
            domain::validate_runtime_domain(child.workload_domain(), workload_domain)?;
            if child.fingerprint() != &fingerprint
                || catalog
                    .values()
                    .any(|old: &ImportedStructuredModelV2| old.same_population(&child))
            {
                return Err(FerrumError::config(
                    "live catalog fingerprint or owner differs",
                ));
            }
            child
                .is_current_local(now)
                .map_err(|e| profile_error(format!("live catalog source is not fresh: {e:?}")))?;
            if catalog.insert(*child.domain_signature(), child).is_some() {
                return Err(FerrumError::config("live catalog duplicate domain"));
            }
        }
        Ok(Arc::new(Self {
            fingerprint,
            inner: Snapshot::StructuredV2(StructuredSnapshot {
                children: Arc::new(catalog),
                feedback: None,
                epoch,
            }),
        }))
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn validate_live_freshness(
        &self,
        now: u64,
    ) -> Result<(), FerrumError> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config("not a structured catalog"));
        };
        for child in value.children.values() {
            child
                .is_current_local(now)
                .map_err(|e| profile_error(format!("catalog expired before publication: {e:?}")))?;
        }
        Ok(())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn validate_live_receipt(
        &self,
        receipt: &SloCostProfileReceipt,
    ) -> Result<(), FerrumError> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config("not a structured catalog"));
        };
        let declared = receipt
            .structured_whole_wave_v2
            .as_ref()
            .ok_or_else(|| FerrumError::config("replacement lacks structured receipt"))?;
        if declared.children.is_empty() || declared.children.len() != declared.child_count {
            return Err(FerrumError::config(
                "replacement receipt population differs",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for record in &declared.children {
            let child = value
                .children
                .get(&record.domain_signature)
                .ok_or_else(|| FerrumError::config("replacement receipt owner missing"))?;
            if !seen.insert(record.domain_signature)
                || serde_json::to_value(record).map_err(profile_error)?
                    != serde_json::to_value(receipt::child(child)?).map_err(profile_error)?
            {
                return Err(FerrumError::config(
                    "replacement receipt source binding differs",
                ));
            }
        }
        Ok(())
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn retained_live_domains(
        &self,
        next: &Self,
    ) -> Vec<[u8; 32]> {
        let (Snapshot::StructuredV2(old), Snapshot::StructuredV2(new)) = (&self.inner, &next.inner)
        else {
            return Vec::new();
        };
        old.children
            .iter()
            .filter_map(|(domain, a)| {
                let b = new.children.get(domain)?;
                let (a, b) = (a.provenance(), b.provenance());
                (a.file_sha256 == b.file_sha256
                    && a.source_sha256 == b.source_sha256
                    && a.parameters_sha256 == b.parameters_sha256
                    && a.protocol == b.protocol
                    && a.capture_identity == b.capture_identity)
                    .then_some(*domain)
            })
            .collect()
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn live_feedback_binding(
        &self,
        policy: &ferrum_types::SloSelectedFeedbackSettingsV1,
    ) -> Result<
        (
            super::super::super::selected_feedback::Binding,
            Arc<[[u8; 32]]>,
        ),
        FerrumError,
    > {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config("not a structured catalog"));
        };
        Ok((
            value.feedback_binding(policy, &self.fingerprint)?,
            value.children.keys().copied().collect::<Vec<_>>().into(),
        ))
    }
}
