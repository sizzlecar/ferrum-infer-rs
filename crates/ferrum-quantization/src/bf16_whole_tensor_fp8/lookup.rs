//! Per-group exact memoization of the existing scalar encoder, not another
//! quantization rule. The finite positive BF16 domain has only 32640 entries.
use super::*;

pub(super) const MAGNITUDE_ENTRIES: usize = 0x7f80;
// Build at most one scalar result per four requested packed output elements.
// This bounds setup work by element counts; it is not a measured crossover.
pub(super) const MINIMUM_PACKED_ELEMENTS: u64 = 4 * MAGNITUDE_ENTRIES as u64;

pub(super) const fn workspace_bytes(packed_elements: u64) -> usize {
    if packed_elements >= MINIMUM_PACKED_ELEMENTS {
        MAGNITUDE_ENTRIES
    } else {
        0
    }
}

pub(super) struct LookupTable {
    bytes: ManagedHostBytes,
    maximum_magnitude: usize,
}

impl LookupTable {
    pub(super) fn new(
        host: &HostMemoryAccount,
        maximum: f32,
        inverse: f32,
    ) -> Result<Self, VNextError> {
        let bits = maximum.to_bits();
        if !maximum.is_finite()
            || bits >> 31 != 0
            || bits & 0xffff != 0
            || !inverse.is_finite()
            || inverse <= 0.0
            || !(maximum * inverse).is_finite()
        {
            return Err(invalid(
                "lookup requires a finite BF16 maximum and reciprocal",
            ));
        }
        let maximum_magnitude = (bits >> 16) as usize;
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, MAGNITUDE_ENTRIES)
            .map_err(|error| invalid(format!("WeightBacking lookup admission: {error}")))?;
        for (magnitude, target) in bytes.as_mut()[..=maximum_magnitude].iter_mut().enumerate() {
            let value = f32::from_bits((magnitude as u32) << 16);
            *target = conversion::encode(value * inverse);
        }
        Ok(Self {
            bytes: bytes.freeze(),
            maximum_magnitude,
        })
    }

    /// The preceding complete amax scan has proved every source magnitude is
    /// finite and no larger than this table's initialized prefix. With a finite
    /// positive reciprocal, RN32 multiplication and the scalar encoder are
    /// sign symmetric, including negative zero, underflow and saturation.
    pub(super) fn encode_into(&self, source: &[u8], target: &mut [u8]) {
        debug_assert_eq!(source.len() % 2, 0);
        debug_assert_eq!(source.len() / 2, target.len());
        // Resolve the managed slice once per matrix rather than relying on
        // cross-crate inlining of ManagedHostBytes::bytes in the element loop.
        let table = self.bytes.bytes();
        for (original, target) in source.chunks_exact(2).zip(target) {
            let bits = u16::from_le_bytes([original[0], original[1]]);
            let magnitude = usize::from(bits & 0x7fff);
            debug_assert!(magnitude <= self.maximum_magnitude);
            *target = table[magnitude] | ((bits >> 8) as u8 & 0x80);
        }
    }

    #[cfg(test)]
    pub(super) fn encode(&self, bits: u16) -> u8 {
        let mut target = [0];
        self.encode_into(&bits.to_le_bytes(), &mut target);
        target[0]
    }
}
