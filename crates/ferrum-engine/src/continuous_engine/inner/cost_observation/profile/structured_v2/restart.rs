//! A cold proof of the same qualified catalog across storage representations.
//! It is private, borrowed and non-deserializable; it grants no execution right.
use super::*;
use crate::continuous_engine::inner::cost_observation::selected_feedback::Binding;
use ferrum_interfaces::execution_cost::CostMonotonicDomainV1;
use ferrum_types::{
    SloCostProfileClockBasis, SloCostProfileStorage, SloSelectedFeedbackSettingsV1,
};
#[cfg(test)]
mod tests;

pub(in crate::continuous_engine::inner::cost_observation) struct VerifiedRestartCatalog<'a> {
    original: &'a EngineCostSnapshot,
    replayed: &'a EngineCostSnapshot,
    before: Binding,
    after: Binding,
    scopes: Arc<[[u8; 32]]>,
    checked_at_ns: u64,
}
impl EngineCostSnapshot {
    /// Copy only bounded immutable origin descriptors. Feedback revocation and
    /// original freshness are checked by the later restart proof; this method
    /// grants neither prediction nor import authority.
    pub(in crate::continuous_engine::inner::cost_observation) fn restart_origins(
        &self,
        output: &mut Vec<super::super::super::automatic_reuse::RestartOrigin>,
    ) -> Result<(), FerrumError> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config(
                "restart inventory requires structured catalog",
            ));
        };
        if value.children.is_empty()
            || value.children.len() > 128
            || output.capacity() < value.children.len()
        {
            return Err(FerrumError::config("restart inventory capacity"));
        }
        for child in value.children.values() {
            let p = child.provenance();
            output.push(super::super::super::automatic_reuse::RestartOrigin {
                capture: p.capture_identity,
                protocol: p.protocol,
                domain: *child.domain_signature(),
                prefix: (p.source_bytes, p.source_sha256),
                offered: p.offered_attempts,
                rows: p.total_shape_rows,
            });
        }
        Ok(())
    }
    /// Original live models coexist with independently replayed models during
    /// a save. Charge their backing before replay allocates a second catalog.
    pub(in crate::continuous_engine::inner::cost_observation) fn restart_retained_bytes(
        &self,
    ) -> Result<usize, FerrumError> {
        let Snapshot::StructuredV2(value) = &self.inner else {
            return Err(FerrumError::config("restart requires structured catalog"));
        };
        value
            .children
            .values()
            .try_fold(std::mem::size_of::<Self>(), |sum, child| {
                sum.checked_add(
                    child.retained_payload_bytes().ok_or_else(|| {
                        FerrumError::config("restart original model size overflow")
                    })?,
                )
                .and_then(|n| n.checked_add(256))
                .ok_or_else(|| FerrumError::config("restart original catalog size overflow"))
            })
    }
    /// Allocation-free cold preflight for the two existing binding encoders
    /// and the one <=128-entry borrowed proof's scope array.
    pub(in crate::continuous_engine::inner::cost_observation) fn restart_verification_budget(
        &self,
        replayed: &Self,
        policy: &SloSelectedFeedbackSettingsV1,
    ) -> Result<usize, FerrumError> {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0 = self
                    .0
                    .checked_add(bytes.len())
                    .ok_or_else(|| std::io::Error::other("restart binding byte overflow"))?;
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        fn encoded(value: &impl serde::Serialize) -> Result<usize, FerrumError> {
            let mut count = Count(0);
            serde_json::to_writer(&mut count, value).map_err(profile_error)?;
            Ok(count.0)
        }
        let mut peak = encoded(policy)?;
        let mut maximum_children = 0;
        for snapshot in [self, replayed] {
            let Snapshot::StructuredV2(value) = &snapshot.inner else {
                return Err(FerrumError::config(
                    "restart proof requires structured catalogs",
                ));
            };
            if value.children.is_empty() || value.children.len() > 128 {
                return Err(FerrumError::config("restart catalog child capacity"));
            }
            maximum_children = maximum_children.max(value.children.len());
            peak = peak.max(encoded(&file::ProfileFingerprint::from(
                &snapshot.fingerprint,
            ))?);
            for child in value.children.values() {
                peak = peak
                    .max(encoded(child.owner())?)
                    .max(encoded(child.scope())?);
            }
        }
        // Each binding encoder releases its Vec before encoding the next
        // field. Account Vec geometric capacity and Vec->Arc scope overlap.
        peak.checked_mul(2)
            .and_then(|n| n.checked_add(maximum_children.checked_mul(64)?))
            .and_then(|n| n.checked_add(std::mem::size_of::<VerifiedRestartCatalog<'_>>()))
            .and_then(|n| n.checked_add(512))
            .ok_or_else(|| FerrumError::config("restart proof memory bound overflow"))
    }

    pub(in crate::continuous_engine::inner::cost_observation) fn verify_restart_replay<'a>(
        &'a self,
        replayed: &'a Self,
        policy: &SloSelectedFeedbackSettingsV1,
        actual_domain: &CostMonotonicDomainV1,
        now: u64,
        maximum_transient_bytes: usize,
    ) -> Result<VerifiedRestartCatalog<'a>, FerrumError> {
        let fail = || FerrumError::config("restart replay changes original qualified catalog");
        if self.restart_verification_budget(replayed, policy)? > maximum_transient_bytes {
            return Err(FerrumError::config("restart proof exceeds reserved memory"));
        }
        let (Snapshot::StructuredV2(old), Snapshot::StructuredV2(new)) =
            (&self.inner, &replayed.inner)
        else {
            return Err(fail());
        };
        if self.fingerprint != replayed.fingerprint
            || old.children.is_empty()
            || old.children.len() > 128
            || old.children.len() != new.children.len()
        {
            return Err(fail());
        }
        for (domain, a) in old.children.iter() {
            let b = new.children.get(domain).ok_or_else(fail)?;
            let (p, q) = (a.provenance(), b.provenance());
            if a.monotonic_domain() != Some(actual_domain)
                || b.monotonic_domain() != Some(actual_domain)
                || p.clock_basis != SloCostProfileClockBasis::SameBootMonotonic
                || q.clock_basis != SloCostProfileClockBasis::SameBootMonotonic
                || q.storage != SloCostProfileStorage::File
                || q.loaded_from.is_none()
                || q.source_path.is_none()
                || a.fingerprint() != b.fingerprint()
                || a.owner() != b.owner()
                || a.scope() != b.scope()
                || a.workload_domain() != b.workload_domain()
                || a.runtime_limits() != b.runtime_limits()
                || a.parameters_signature() != b.parameters_signature()
                || p.schema_version != q.schema_version
                || p.source_sha256 != q.source_sha256
                || p.source_bytes != q.source_bytes
                || p.parameters_sha256 != q.parameters_sha256
                || p.capture_identity != q.capture_identity
                || p.protocol != q.protocol
                || p.rule_signature != q.rule_signature
                || p.cohort_manifest_sha256 != q.cohort_manifest_sha256
                || p.offered_attempts != q.offered_attempts
                || p.reserved_members != q.reserved_members
                || p.total_shape_rows != q.total_shape_rows
                || p.phases != q.phases
                || p.clock.source_monotonic_anchor_ns != q.clock.source_monotonic_anchor_ns
                || p.clock.model_anchor_ns != q.clock.model_anchor_ns
                || p.conservative_clock_error_ns != 0
                || q.conservative_clock_error_ns != 0
            {
                return Err(fail());
            }
            a.is_current_local(now).map_err(|_| fail())?;
            b.is_current_local(now).map_err(|_| fail())?;
        }
        let before = old.feedback_binding(policy, &self.fingerprint)?;
        let after = new.feedback_binding(policy, &replayed.fingerprint)?;
        // Only metadata's file identity may change, not the population's
        // original source, numerical parameters, protocol or policy digest.
        if before.source_sha256 != after.source_sha256
            || before.fit_sha256 != after.fit_sha256
            || before.protocol_sha256 != after.protocol_sha256
            || before.policy_sha256 != after.policy_sha256
        {
            return Err(fail());
        }
        Ok(VerifiedRestartCatalog {
            original: self,
            replayed,
            before,
            after,
            scopes: old.children.keys().copied().collect::<Vec<_>>().into(),
            checked_at_ns: now,
        })
    }
}
impl VerifiedRestartCatalog<'_> {
    pub(in crate::continuous_engine::inner::cost_observation) fn original_binding(
        &self,
    ) -> &Binding {
        &self.before
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn persisted_binding(
        &self,
    ) -> &Binding {
        &self.after
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn scopes(&self) -> &Arc<[[u8; 32]]> {
        &self.scopes
    }
    pub(in crate::continuous_engine::inner::cost_observation) fn validate_freshness(
        &self,
        now: u64,
    ) -> Result<(), FerrumError> {
        if now < self.checked_at_ns {
            return Err(FerrumError::config("restart clock moved backwards"));
        }
        self.original.validate_live_freshness(now)?;
        self.replayed.validate_live_freshness(now)
    }
}
