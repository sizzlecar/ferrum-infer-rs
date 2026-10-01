//! Input-only native preparation contract, frozen before original samples.
use super::*;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredNativePrefixAcquisitionCohortV1 {
    pub prompt_tokens: u64,
    pub boundary_tokens: u64,
    pub input_tokens_sha256: [u8; 32],
    pub native_scope: StructuredNativePrefixScopeV1,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredNativePrefixScopeV1 {
    pub plan_hash: String,
    pub layout_fingerprint: String,
    pub runtime_implementation_fingerprint: String,
    pub device_id: String,
}
impl StructuredNativePrefixScopeV1 {
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        self.plan_hash
            .capacity()
            .checked_add(self.layout_fingerprint.capacity())?
            .checked_add(self.runtime_implementation_fingerprint.capacity())?
            .checked_add(self.device_id.capacity())
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredNativePrefixAcquisitionPlanV1 {
    pub phases: [Vec<Option<StructuredNativePrefixAcquisitionCohortV1>>; 3],
}
impl StructuredNativePrefixAcquisitionPlanV1 {
    pub fn validate(
        &self,
        cohorts: &CohortPlanV2,
        prefix: &crate::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixPlanV5,
    ) -> Result<(), CostProfileError> {
        let mut any = false;
        for phase in 0..3 {
            if self.phases[phase].len() != cohorts.phases[phase].len()
                || self.phases[phase].len() != prefix.phases[phase].len()
            {
                return Err(invalid(
                    "native acquisition phase/cohort declaration differs",
                ));
            }
            for (ordinal, item) in self.phases[phase].iter().enumerate() {
                if let Some(item) = item {
                    any = true;
                    if prefix.phases[phase][ordinal].is_none()
                        || item.boundary_tokens == 0
                        || item.boundary_tokens >= item.prompt_tokens
                        || item.prompt_tokens > u64::from(u32::MAX)
                        || item.input_tokens_sha256 == [0; 32]
                        || [
                            &item.native_scope.plan_hash,
                            &item.native_scope.layout_fingerprint,
                            &item.native_scope.runtime_implementation_fingerprint,
                            &item.native_scope.device_id,
                        ]
                        .iter()
                        .any(|value| value.is_empty())
                    {
                        return Err(invalid(
                            "native acquisition requires a declared proper prefix",
                        ));
                    }
                }
            }
        }
        if !any {
            return Err(invalid("native acquisition declaration has no acquisition"));
        }
        Ok(())
    }
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        let mut bytes = 0usize;
        for phase in &self.phases {
            bytes = bytes.checked_add(phase.capacity().checked_mul(std::mem::size_of::<
                Option<StructuredNativePrefixAcquisitionCohortV1>,
            >())?)?;
            for item in phase.iter().flatten() {
                bytes = bytes.checked_add(item.native_scope.retained_heap_bytes()?)?;
            }
        }
        Some(bytes)
    }
}
