//! Checked cross-width numerical identity, with no model/publication authority.
//! The original owner, recipe binding and coordinates remain unchanged.
//! A serialized key is a numerical DTO, never a checked input or authority.
use super::*;
use ferrum_interfaces::execution_cost::{ActualWaveKind, CostWorkloadDomainV1, HostCostPolicyV2};
use sha2::{Digest, Sha256};

/// One homogeneous ordinary-decode numerical family. Its algorithm roster is
/// exact: absent/new algorithms are not unioned or silently assigned zero work.
/// Deserialization only reconstructs a comparison value. Consumers must compare
/// it with a key obtained from an independently checked input; it cannot attach
/// a physical domain, change an owner, or authorize execution/publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalFamilyKeyV1 {
    workload_domain: [u8; 32],
    algorithm_domain: [u8; 32],
    host_policy: HostCostPolicyV2,
    product: StructuredProductV2,
    readback: CoreReadbackRoute,
    route: [u64; 5],
    basis_axes: usize,
    support_axes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    algorithm_universe: Option<[u8; 32]>,
}

impl StructuredInputV2 {
    /// Input-only catalogue lookup. Uses cached signatures and checked roster
    /// metadata; no vector allocation, serialization, digest, or recipe replay.
    pub fn numerical_family_key_for_universe(
        &self,
        universe: &DeclaredAlgorithmUniverseV1,
    ) -> Result<NumericalFamilyKeyV1> {
        let mut key = NumericalFamilyKeyV1::from_validated_input(self)?;
        let (basis, support) = universe.checked_axis_counts(self)?;
        key.algorithm_domain = *universe.signature();
        key.algorithm_universe = Some(*universe.signature());
        key.basis_axes = basis;
        key.support_axes = support;
        Ok(key)
    }
}

impl NumericalFamilyKeyV1 {
    pub(super) fn from_validated_input(input: &StructuredInputV2) -> Result<Self> {
        let workload_domain = input
            .physical_domain
            .ok_or(StructuredUnknown::WrongDomain)?;
        if input.template_route_facts[0] != ActualWaveKind::Decode as u64
            || input.owner.role != StructuredWaveRoleV2::OrdinaryDecode
        {
            return Err(StructuredUnknown::UnsupportedScope);
        }
        let host_policy = input
            .homogeneous_host_policy
            .ok_or(StructuredUnknown::UnsupportedScope)?;
        let [kind, path, graph, row_order, _, retries] = input.template_route_facts;
        Ok(Self {
            workload_domain,
            algorithm_domain: input
                .algorithm_universe
                .as_ref()
                .map_or(input.owner.algorithm_domain, |u| *u.signature()),
            host_policy,
            product: input.owner.product,
            readback: input.owner.readback,
            route: [kind, path, graph, row_order, retries],
            basis_axes: input.basis.len(),
            support_axes: input.support.len(),
            algorithm_universe: input.algorithm_universe.as_ref().map(|u| *u.signature()),
        })
    }

    pub fn workload_domain_signature(&self) -> &[u8; 32] {
        &self.workload_domain
    }

    pub fn basis_axes(&self) -> usize {
        self.basis_axes
    }

    pub fn support_axes(&self) -> usize {
        self.support_axes
    }

    /// Cold serialization identity for frozen evidence. Hot membership compares
    /// this typed key directly. Preserve the original family transcript bytes.
    pub fn signature(&self) -> Result<[u8; 32]> {
        let mut digest = Sha256::new();
        digest.update(b"ferrum.homogeneous-decode-numerical-family.v1\0");
        digest.update(MODEL_REVISION_V2.as_bytes());
        digest.update(self.workload_domain);
        digest.update(self.algorithm_domain);
        digest.update(
            serde_json::to_vec(&(self.host_policy, self.product, self.readback))
                .map_err(|_| StructuredUnknown::InvalidInput)?,
        );
        for value in self.route {
            digest.update(value.to_le_bytes());
        }
        digest.update((self.basis_axes as u64).to_le_bytes());
        digest.update((self.support_axes as u64).to_le_bytes());
        if let Some(universe) = self.algorithm_universe {
            digest.update(b"ferrum.declared-algorithm-universe.projection.v1\0");
            digest.update(universe);
        }
        Ok(digest.finalize().into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NumericalFamilyV1 {
    key: NumericalFamilyKeyV1,
    signature: [u8; 32],
}

impl NumericalFamilyV1 {
    /// Versioned numerical identity only; never an execution/profile witness.
    pub fn signature(&self) -> &[u8; 32] {
        &self.signature
    }
}

/// Original checked input paired with a separate, width-independent identity.
/// Declaring an executor fingerprint in a workload domain is not attestation:
/// a future source/collector adapter must still validate that original source.
#[derive(Debug, Clone)]
pub struct NumericalFamilyInputV1 {
    family: NumericalFamilyV1,
    original: StructuredInputV2,
}

impl NumericalFamilyInputV1 {
    pub fn from_actual(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        domain: &CostWorkloadDomainV1,
    ) -> Result<Self> {
        // Includes original command/shape binding and actual row/state limits.
        // In particular, total recurrent bytes must still equal rows * the
        // declared per-row amount before the numerical identity can omit rows.
        let original = StructuredInputV2::from_actual_with_domain(exact, selected, recipe, domain)?;
        let key = original.numerical_family_key()?;
        let family = NumericalFamilyV1 {
            signature: key.signature()?,
            key,
        };
        Ok(Self { family, original })
    }

    pub fn family(&self) -> &NumericalFamilyV1 {
        &self.family
    }

    /// Retains the exact rows, original owner/domain and all physical host facts.
    /// In particular this does not make two inputs share an old owner/profile.
    pub fn original_input(&self) -> &StructuredInputV2 {
        &self.original
    }

    pub fn regression_axes(&self) -> &[f64] {
        self.original.regression_axes()
    }

    /// Checked membership for a future numerical aggregation adapter. This
    /// grants no Fit/Residual/Qualification or source-population authority.
    pub fn same_family(&self, other: &Self) -> Result<()> {
        if self.family != other.family {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
}
