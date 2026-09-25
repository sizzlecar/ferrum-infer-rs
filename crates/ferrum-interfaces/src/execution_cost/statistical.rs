//! Versioned, passive selected-algorithm/work evidence beside exact canonical
//! execution. No statistical identity authorizes work or implies a cost model.
//! Default builders retain bounded stack state. Explicit capture may retain a
//! bounded sparse algorithm-work table, never a second execution trace.
use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

mod algorithm_work;
mod command;
mod independent_rows;
mod structure;
mod wave;
mod wave_algorithm_work;
mod wire;
pub use algorithm_work::{
    AlgorithmNumericWorkV1, AlgorithmWorkKindV1, SelectedAlgorithmWorkEvidenceV1,
};
pub use command::*;
pub(in crate::execution_cost) use structure::StructuredHostAccumulator;
pub use structure::*;
pub(in crate::execution_cost) use wave::StatisticalWaveAccumulator;
pub use wave::{
    CanonicalStatisticalWave, IndependentAttentionWaveEvidenceV2, StatisticalWaveEvidenceV1,
};
pub use wave_algorithm_work::{DeviceAlgorithmWorkEvidenceV1, WaveAlgorithmNumericWorkV1};
pub use wire::{IndependentAttentionWaveEvidenceWireV2, StatisticalWaveEvidenceWireV1};
#[cfg(test)]
mod tests;

pub const STATISTICAL_ROUTE_WORK_SCHEMA_V1: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalEvidenceUnknown {
    MissingProducer,
    InvalidAlgorithm,
    InvalidWork,
    CommandMismatch,
    ExactBindingMismatch,
    MissingHostDomain,
    UnsupportedWave,
    UnsupportedReplay,
    Capacity,
    Overflow,
}

fn number(hash: &mut Sha256, value: u64) {
    hash.update(value.to_le_bytes());
}
fn bytes(hash: &mut Sha256, value: &[u8]) {
    number(hash, value.len() as u64);
    hash.update(value);
}
fn text_valid(text: &str) -> bool {
    !text.is_empty() && text.len() <= 1024
}

/// A provider must derive this from the SAME selected launch used to encode
/// and to project. The operation display label alone is not an algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SelectedAlgorithmClassV1([u8; 32]);
impl SelectedAlgorithmClassV1 {
    pub fn new(
        kernel_entry: &str,
        abi_revision: u32,
        numerical_signature: [u8; 32],
        layout_tile_reduction_signature: [u8; 32],
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if !text_valid(kernel_entry)
            || abi_revision == 0
            || numerical_signature == [0; 32]
            || layout_tile_reduction_signature == [0; 32]
        {
            return Err(StatisticalEvidenceUnknown::InvalidAlgorithm);
        }
        let mut hash = Sha256::new();
        bytes(&mut hash, b"ferrum.selected-algorithm-class.v1");
        bytes(&mut hash, kernel_entry.as_bytes());
        number(&mut hash, u64::from(abi_revision));
        hash.update(numerical_signature);
        hash.update(layout_tile_reduction_signature);
        Ok(Self(hash.finalize().into()))
    }
    pub fn signature(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Real launch quantities, not FLOPs or latency bounds. The meaning of one
/// logical unit is fixed by SelectedAlgorithmClassV1. No address or owner ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelNumericWorkV1 {
    pub logical_units: u64,
    pub padded_units: u64,
    pub inner_units_per_logical_unit: u64,
    pub grid: [u32; 3],
    pub scratch_bytes: u64,
    pub staged_weight_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalTransferKindV1 {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Fill,
}

/// All sums use checked arithmetic; peak scratch is a maximum, never the sum
/// of overlapping workspace allocations. No kernel elapsed times are summed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct DeviceNumericWorkV1 {
    pub logical_units: u64,
    pub padded_units: u64,
    pub inner_work_units: u64,
    pub grid_blocks: u64,
    pub peak_scratch_bytes: u64,
    pub staged_weight_bytes: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub device_to_device_bytes: u64,
    pub fill_bytes: u64,
}
impl DeviceNumericWorkV1 {
    fn checked_add(self, other: Self) -> Result<Self, StatisticalEvidenceUnknown> {
        use StatisticalEvidenceUnknown::Overflow;
        Ok(Self {
            logical_units: self
                .logical_units
                .checked_add(other.logical_units)
                .ok_or(Overflow)?,
            padded_units: self
                .padded_units
                .checked_add(other.padded_units)
                .ok_or(Overflow)?,
            inner_work_units: self
                .inner_work_units
                .checked_add(other.inner_work_units)
                .ok_or(Overflow)?,
            grid_blocks: self
                .grid_blocks
                .checked_add(other.grid_blocks)
                .ok_or(Overflow)?,
            peak_scratch_bytes: self.peak_scratch_bytes.max(other.peak_scratch_bytes),
            staged_weight_bytes: self
                .staged_weight_bytes
                .checked_add(other.staged_weight_bytes)
                .ok_or(Overflow)?,
            host_to_device_bytes: self
                .host_to_device_bytes
                .checked_add(other.host_to_device_bytes)
                .ok_or(Overflow)?,
            device_to_host_bytes: self
                .device_to_host_bytes
                .checked_add(other.device_to_host_bytes)
                .ok_or(Overflow)?,
            device_to_device_bytes: self
                .device_to_device_bytes
                .checked_add(other.device_to_device_bytes)
                .ok_or(Overflow)?,
            fill_bytes: self
                .fill_bytes
                .checked_add(other.fill_bytes)
                .ok_or(Overflow)?,
        })
    }
}
