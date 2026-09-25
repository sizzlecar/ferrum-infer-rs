//! Shared numerical owner facts. These do not mint execution or live receipts.
use super::*;
use ferrum_interfaces::execution_cost::{
    AlgorithmWorkKindV1, HostContentDomainV1, HostCostPolicyV2, HostRowRoleV2,
    StructuredCostProductV1,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize)]
pub struct StructuredOwnerFactsV2 {
    rows: Vec<StructuredOwnerRowFactV2>,
    product: StructuredProductV2,
    readback: CoreReadbackRoute,
    provider_template: StructuredTemplateV2,
    algorithms: Vec<StructuredAlgorithmFactV2>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredOwnerRowFactV2 {
    pub role: HostRowRoleV2,
    pub no_generated_history: bool,
    pub installed_policy: HostCostPolicyV2,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredAlgorithmFactV2 {
    pub signature: [u8; 32],
    pub kind: AlgorithmWorkKindV1,
}
impl StructuredOwnerFactsV2 {
    /// Numerical replay only. The importer validates the complete original
    /// ledger; this helper cannot mint a live recipe or settlement receipt.
    pub(in crate::implementations::continuous) fn from_replay_parts(
        rows: &[StructuredHostRowV1],
        product: StructuredProductV2,
        readback: CoreReadbackRoute,
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        algorithms: &[(
            [u8; 32],
            AlgorithmWorkKindV1,
            u64,
            ferrum_interfaces::execution_cost::DeviceNumericWorkV1,
        )],
    ) -> Result<Self> {
        if rows.is_empty() || rows.len() > 128 || algorithms.is_empty() || algorithms.len() > 4096 {
            return Err(StructuredUnknown::Capacity);
        }
        let value = Self {
            rows: rows
                .iter()
                .map(|r| StructuredOwnerRowFactV2 {
                    role: r.role,
                    no_generated_history: r.no_generated_history,
                    installed_policy: r.installed_policy,
                })
                .collect(),
            product,
            readback,
            provider_template: grouped
                .map(StructuredTemplateV2::ProviderGrouped)
                .unwrap_or(StructuredTemplateV2::Ordered(ordered)),
            algorithms: algorithms
                .iter()
                .map(|(signature, kind, _, _)| StructuredAlgorithmFactV2 {
                    signature: *signature,
                    kind: *kind,
                })
                .collect(),
        };
        value.owner_key()?;
        Ok(value)
    }
    pub fn from_prepared(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) -> Result<Self> {
        let attached = selected
            .structured_capture()
            .ok_or(StructuredUnknown::MissingEvidence)?
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        if !Arc::ptr_eq(attached, recipe) {
            return Err(StructuredUnknown::MissingEvidence);
        }
        recipe
            .validate_exact(exact)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let work = recipe
            .algorithm_work()
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        work.validate_structure(recipe)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        // Validate the original exact/statistical bridge as well as the sidecar.
        super::super::statistical::StatisticalModelInputV1::from_future(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let device = recipe.device();
        let value = Self {
            rows: recipe
                .physical_host_rows()
                .iter()
                .map(|r| StructuredOwnerRowFactV2 {
                    role: r.role,
                    no_generated_history: r.no_generated_history,
                    installed_policy: r.installed_policy,
                })
                .collect(),
            product: match device.product() {
                StructuredCostProductV1::GreedyToken => StructuredProductV2::GreedyToken,
                StructuredCostProductV1::FullLogits => StructuredProductV2::FullLogits,
            },
            readback: device.readback(),
            provider_template: device
                .provider_grouped_template()
                .copied()
                .map(StructuredTemplateV2::ProviderGrouped)
                .unwrap_or(StructuredTemplateV2::Ordered(*device.ordered_template())),
            algorithms: work
                .entries()
                .iter()
                .map(|a| StructuredAlgorithmFactV2 {
                    signature: *a.algorithm().signature(),
                    kind: a.kind(),
                })
                .collect(),
        };
        value.owner_key()?;
        Ok(value)
    }
    pub fn physical_rows(&self) -> &[StructuredOwnerRowFactV2] {
        &self.rows
    }
    pub fn owner_key(&self) -> Result<StructuredOwnerKeyV2> {
        if self.rows.is_empty()
            || self.rows.len() > 128
            || self.algorithms.is_empty()
            || self.algorithms.len() > 4096
            || self.readback == CoreReadbackRoute::Unknown
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        let template = match self.provider_template {
            StructuredTemplateV2::Ordered(v) | StructuredTemplateV2::ProviderGrouped(v) => v,
        };
        if template == [0; 32] {
            return Err(StructuredUnknown::MissingEvidence);
        }
        let mut decode = 0;
        let mut prefill = 0;
        let mut first_decode = false;
        let mut policy = Sha256::new();
        policy.update(b"ferrum.structured-owner-policies.v2\0");
        for row in &self.rows {
            match row.role {
                HostRowRoleV2::Decode => {
                    decode += 1;
                    first_decode |= row.no_generated_history;
                }
                HostRowRoleV2::Prefill => prefill += 1,
            }
            let p = row.installed_policy;
            if p.empirical_content_domain != Some(HostContentDomainV1::PlainTextGreedyV1)
                || p.categorical_signature == [0; 32]
                || p.decoder_text_bytes_per_token == 0
                || p.raw_token_bytes_bound == 0
                || [
                    p.decoder_text_bytes_per_token,
                    p.decoder_scratch_bytes_per_token,
                    p.raw_token_bytes_bound,
                ]
                .iter()
                .any(|v| *v > (1 << 53))
            {
                return Err(StructuredUnknown::UnsupportedScope);
            }
            policy.update(p.categorical_signature);
            for value in [
                p.decoder_text_bytes_per_token,
                p.decoder_scratch_bytes_per_token,
                p.raw_token_bytes_bound,
            ] {
                policy.update(value.to_le_bytes());
            }
        }
        let role = match (decode, prefill, first_decode) {
            (0, _, _) => StructuredWaveRoleV2::Prefill,
            (_, 0, false) => StructuredWaveRoleV2::OrdinaryDecode,
            (_, 0, true) => StructuredWaveRoleV2::DecodeWithNoGeneratedHistory,
            _ => StructuredWaveRoleV2::Mixed,
        };
        let mut algorithm = Sha256::new();
        algorithm.update(b"ferrum.structured-owner-algorithms.v2\0");
        let mut previous = None;
        for a in &self.algorithms {
            let key = (a.signature, a.kind);
            if a.signature == [0; 32] || previous.is_some_and(|p| p >= key) {
                return Err(StructuredUnknown::InvalidInput);
            }
            previous = Some(key);
            algorithm.update(a.signature);
            algorithm.update([match a.kind {
                AlgorithmWorkKindV1::Kernel => 0,
                AlgorithmWorkKindV1::HostToDevice => 1,
                AlgorithmWorkKindV1::DeviceToHost => 2,
                AlgorithmWorkKindV1::DeviceToDevice => 3,
                AlgorithmWorkKindV1::Fill => 4,
                AlgorithmWorkKindV1::LibraryCall => 5,
            }]);
        }
        Ok(StructuredOwnerKeyV2 {
            rows: self.rows.len() as u32,
            role,
            product: self.product,
            readback: self.readback,
            provider_template: self.provider_template,
            algorithm_domain: algorithm.finalize().into(),
            installed_policy: policy.finalize().into(),
        })
    }
}
