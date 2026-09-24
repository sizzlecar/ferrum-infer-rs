//! Only an explicitly versioned new profile may carry this record. Decoding
//! metadata does not establish execution authority or invent old observations.
use super::*;
use serde::Deserialize;
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatisticalWaveEvidenceWireV1 {
    schema_version: u32,
    exact_binding: [u8; 32],
    family_signature: [u8; 32],
    physical_commands: u32,
    work: Work,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Work {
    logical_units: u64,
    padded_units: u64,
    inner_work_units: u64,
    grid_blocks: u64,
    peak_scratch_bytes: u64,
    staged_weight_bytes: u64,
    host_to_device_bytes: u64,
    device_to_host_bytes: u64,
    device_to_device_bytes: u64,
    fill_bytes: u64,
}
impl From<DeviceNumericWorkV1> for Work {
    fn from(v: DeviceNumericWorkV1) -> Self {
        Self {
            logical_units: v.logical_units,
            padded_units: v.padded_units,
            inner_work_units: v.inner_work_units,
            grid_blocks: v.grid_blocks,
            peak_scratch_bytes: v.peak_scratch_bytes,
            staged_weight_bytes: v.staged_weight_bytes,
            host_to_device_bytes: v.host_to_device_bytes,
            device_to_host_bytes: v.device_to_host_bytes,
            device_to_device_bytes: v.device_to_device_bytes,
            fill_bytes: v.fill_bytes,
        }
    }
}
impl From<Work> for DeviceNumericWorkV1 {
    fn from(v: Work) -> Self {
        Self {
            logical_units: v.logical_units,
            padded_units: v.padded_units,
            inner_work_units: v.inner_work_units,
            grid_blocks: v.grid_blocks,
            peak_scratch_bytes: v.peak_scratch_bytes,
            staged_weight_bytes: v.staged_weight_bytes,
            host_to_device_bytes: v.host_to_device_bytes,
            device_to_host_bytes: v.device_to_host_bytes,
            device_to_device_bytes: v.device_to_device_bytes,
            fill_bytes: v.fill_bytes,
        }
    }
}
impl StatisticalWaveEvidenceV1 {
    pub fn to_wire_v1(&self) -> StatisticalWaveEvidenceWireV1 {
        StatisticalWaveEvidenceWireV1 {
            schema_version: self.schema_version,
            exact_binding: self.exact_binding,
            family_signature: self.family_signature,
            physical_commands: self.physical_commands,
            work: self.work.into(),
        }
    }
    pub fn from_wire_v1(
        wire: StatisticalWaveEvidenceWireV1,
        exact: &CanonicalWaveCostShape,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if wire.family_signature == [0; 32]
            || wire.work.padded_units < wire.work.logical_units
            || (wire.work.logical_units > 0
                && (wire.work.inner_work_units == 0 || wire.work.grid_blocks == 0))
        {
            return Err(StatisticalEvidenceUnknown::InvalidWork);
        }
        let value = Self {
            schema_version: wire.schema_version,
            exact_binding: wire.exact_binding,
            family_signature: wire.family_signature,
            physical_commands: wire.physical_commands,
            work: wire.work.into(),
            independent_attention_v2: None,
        };
        value.validate_exact(exact)?;
        Ok(value)
    }
}

/// Only explicit new capture/profile protocols may publish this sidecar. A V1
/// ordered digest cannot reconstruct it, even if all row counters are known.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndependentAttentionWaveEvidenceWireV2 {
    schema_version: u32,
    exact_binding: [u8; 32],
    family_signature: [u8; 32],
    physical_commands: u32,
    work: Work,
}
impl IndependentAttentionWaveEvidenceV2 {
    pub fn to_wire_v2(&self) -> IndependentAttentionWaveEvidenceWireV2 {
        IndependentAttentionWaveEvidenceWireV2 {
            schema_version: self.schema_version,
            exact_binding: self.exact_binding,
            family_signature: self.family_signature,
            physical_commands: self.physical_commands,
            work: self.work.into(),
        }
    }
    pub fn from_wire_v2(
        wire: IndependentAttentionWaveEvidenceWireV2,
        exact: &CanonicalWaveCostShape,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if wire.family_signature == [0; 32]
            || wire.work.padded_units < wire.work.logical_units
            || (wire.work.logical_units > 0
                && (wire.work.inner_work_units == 0 || wire.work.grid_blocks == 0))
        {
            return Err(StatisticalEvidenceUnknown::InvalidWork);
        }
        let value = Self {
            schema_version: wire.schema_version,
            exact_binding: wire.exact_binding,
            family_signature: wire.family_signature,
            physical_commands: wire.physical_commands,
            work: wire.work.into(),
        };
        value.validate_exact(exact)?;
        Ok(value)
    }
}
