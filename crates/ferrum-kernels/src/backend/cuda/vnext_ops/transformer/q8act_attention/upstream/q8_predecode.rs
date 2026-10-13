//! Plan-owned exact F32 destinations for small, untransformed Q8 leaves.

use super::{add, flag_bytes, mul, weights, PreparedProjectionNumerics};
use crate::gguf_blocks::GgufBlockFormat;
use ferrum_interfaces::vnext::WeightEncoding;
use ferrum_types::Q8ProjectionWeightPreparation;

/// This policy is about a physical leaf, including leaves of fused projections.
/// Large projections retain their compressed implementation and memory demand.
const MAX_OUTPUTS: u64 = 64;

pub(super) struct PersistentLayout {
    pub bytes: u64,
    destinations: Vec<Option<Destination>>,
}

#[derive(Clone, Copy)]
pub(super) struct Destination {
    pub offset: u64,
    pub bytes: u64,
    pub inputs: u32,
    pub outputs: u32,
    output_offset: u32,
}

impl PersistentLayout {
    pub fn new(
        numerics: &PreparedProjectionNumerics,
        policy: Q8ProjectionWeightPreparation,
    ) -> Result<Self, String> {
        let mut result = Self {
            bytes: flag_bytes(numerics)?,
            destinations: Vec::new(),
        };
        for projection in numerics.projections() {
            for leaf in projection.leaves() {
                let eligible = policy == Q8ProjectionWeightPreparation::PredecodedF32
                    && !leaf.has_weight_transform()
                    && !leaf.is_staged()
                    && matches!(leaf.encoding(), WeightEncoding::BlockQuantized(spec)
                        if GgufBlockFormat::from_spec(spec) == Ok(GgufBlockFormat::Q8_0));
                let destination = if eligible {
                    Destination::append(
                        &mut result.bytes,
                        projection.input_features(),
                        leaf.output_features(),
                        leaf.output_offset(),
                    )?
                } else {
                    None
                };
                result.destinations.push(destination);
            }
        }
        Ok(result)
    }

    pub fn destination(
        &self,
        first_leaf: u64,
        index: usize,
    ) -> Result<Option<Destination>, String> {
        let index = usize::try_from(add(first_leaf, index as u64)?)
            .map_err(|_| "Q8 preparation leaf index exceeds usize")?;
        self.destinations
            .get(index)
            .copied()
            .ok_or_else(|| "Q8 preparation layout does not cover the actual projection leaf".into())
    }
}

impl Destination {
    fn append(
        cursor: &mut u64,
        inputs: u64,
        outputs: u64,
        output_offset: u64,
    ) -> Result<Option<Self>, String> {
        if inputs == 0 || !inputs.is_multiple_of(32) || !(1..=MAX_OUTPUTS).contains(&outputs) {
            return Ok(None);
        }
        let inputs = u32::try_from(inputs).map_err(|_| "Q8 predecode K exceeds launch ABI")?;
        let outputs = u32::try_from(outputs).map_err(|_| "Q8 predecode N exceeds launch ABI")?;
        let output_offset = u32::try_from(output_offset)
            .map_err(|_| "Q8 predecode output offset exceeds launch ABI")?;
        let offset = add(*cursor, 15)? / 16 * 16;
        let bytes = mul(mul(u64::from(inputs), u64::from(outputs))?, 4)?;
        *cursor = add(offset, bytes)?;
        Ok(Some(Self {
            offset,
            bytes,
            inputs,
            outputs,
            output_offset,
        }))
    }

    pub fn validate(&self, part: &weights::MatrixPart, inputs: u64) -> Result<(), String> {
        if part.format != weights::MatrixFormat::Block(GgufBlockFormat::Q8_0)
            || part.transform.is_some()
            || part.signs_region.is_some()
            || part.columns != self.inputs
            || inputs != u64::from(self.inputs)
            || part.rows != self.outputs
            || part.output_offset != self.output_offset
            || self.offset % 16 != 0
            || self.bytes != mul(mul(u64::from(self.inputs), u64::from(self.outputs))?, 4)?
        {
            return Err("Q8 predecode destination differs from the actual physical leaf".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_projection_weight_preparation_budgets_complete_aligned_leaves() {
        let mut bytes = 24;
        let first = Destination::append(&mut bytes, 5120, 48, 16384)
            .unwrap()
            .unwrap();
        let second = Destination::append(&mut bytes, 5120, 48, 16432)
            .unwrap()
            .unwrap();
        assert_eq!(first.offset, 32);
        assert_eq!(first.bytes, 5120 * 48 * 4);
        assert_eq!(second.offset, first.offset + first.bytes);
        assert_eq!(bytes, 32 + 2 * 5120 * 48 * 4);
        let mut one = 0;
        assert_eq!(
            Destination::append(&mut one, 32, 64, 0)
                .unwrap()
                .unwrap()
                .bytes,
            8192
        );
    }

    #[test]
    fn q8_projection_weight_preparation_rejects_invalid_or_unbounded_extents() {
        for (k, n) in [(0, 48), (31, 48), (32, 0), (32, 65), (5120, 1024)] {
            let mut bytes = 24;
            assert!(Destination::append(&mut bytes, k, n, 0).unwrap().is_none());
            assert_eq!(bytes, 24);
        }
        for (cursor, k, n, offset) in [
            (u64::MAX, 32, 1, 0),
            (0, u64::from(u32::MAX) + 1, 1, 0),
            (0, 32, 1, u64::from(u32::MAX) + 1),
        ] {
            let mut bytes = cursor;
            assert!(Destination::append(&mut bytes, k, n, offset).is_err());
            assert_eq!(bytes, cursor);
        }
    }
}
