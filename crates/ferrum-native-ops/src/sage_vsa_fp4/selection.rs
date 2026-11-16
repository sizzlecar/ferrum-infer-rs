//! GPU region selection metadata. This proof covers geometry, not GPU scores.
//! Dynamic lists are private writes of the selection/attention command; no
//! arbitrary device I32 operand is interpreted as a validated sparse pattern.
use super::*;

mod kernel_upload;
pub use kernel_upload::{RegionKernelPlanV1, REGION_KERNEL_EXPORTS, REGION_PACKET_BYTES};

pub const REGION_SELECTION_PROFILE: u32 = 1;
pub const REGION_SELECTION_NUMERICAL_IDENTITY: &str = concat!(
    "region-topk.f32.ieee-radix-descending-index-ascending.v1;",
    "signed-zero-ordered;NaN-first-payload-independent;infinities-ordered;",
    "prefix-exempt;regions-independent;indices-descending;",
    "torch-cutoff-tie-set-equivalence-not-established"
);
pub const REGION_SELECTION_EXPORTS: [&str; 3] = [
    "ferrum_sage_vsa_fp4_region_plan_v1",
    "ferrum_sage_vsa_fp4_region_validate_v1",
    "ferrum_sage_vsa_fp4_region_run_v1",
];

/// Regions are flat `[start, end, keep]` triples. They cover every video
/// key exactly once, immediately after the nonempty dense prefix. Query
/// blocks are precisely those video blocks. Half-valid counts cover all keys.
/// No allocation, Host registration or device admission happens here.
#[derive(Clone, Copy, Debug)]
pub struct CheckedRegionSelection<'a> {
    shape: SageVsaFp4ListShape,
    regions: &'a [i32],
    half_valid: &'a [i32],
    selected_per_row: u32,
}
impl<'a> CheckedRegionSelection<'a> {
    pub fn new(
        shape: SageVsaFp4ListShape,
        regions: &'a [i32],
        half_valid: &'a [i32],
    ) -> Result<Self, SageVsaFp4Error> {
        let prefix = shape.key_blocks() - shape.query_blocks();
        if prefix == 0
            || regions.is_empty()
            || regions.len() % 3 != 0
            || regions.len() / 3 > shape.query_blocks() as usize
            || half_valid.len() != shape.key_blocks() as usize * 2
        {
            return Err(SageVsaFp4Error::Lists);
        }
        for pair in half_valid.chunks_exact(2) {
            if !(1..=64).contains(&pair[0])
                || !(0..=64).contains(&pair[1])
                || (pair[0] < 64 && pair[1] != 0)
            {
                return Err(SageVsaFp4Error::Lists);
            }
        }
        let mut end = prefix as i32;
        let mut selected = prefix;
        for region in regions.chunks_exact(3) {
            let [start, next, keep] = *region else {
                unreachable!()
            };
            if start != end
                || next <= start
                || next > shape.key_blocks() as i32
                || keep < 1
                || keep > next - start
            {
                return Err(SageVsaFp4Error::Lists);
            }
            end = next;
            selected += keep as u32; // disjoint coverage bounds the sum by K
        }
        if end != shape.key_blocks() as i32 {
            return Err(SageVsaFp4Error::Lists);
        }
        Ok(Self {
            shape,
            regions,
            half_valid,
            selected_per_row: selected,
        })
    }
    pub const fn shape(self) -> SageVsaFp4ListShape {
        self.shape
    }
    pub const fn prefix_blocks(self) -> u32 {
        self.shape.key_blocks() - self.shape.query_blocks()
    }
    pub const fn selected_per_row(self) -> u32 {
        self.selected_per_row
    }
    pub fn regions(self) -> &'a [i32] {
        self.regions
    }
    pub fn half_valid(self) -> &'a [i32] {
        self.half_valid
    }
    pub fn host_addresses(self) -> RegionHostV1 {
        let span = |v: &[i32]| SageVsaFp4SpanV1 {
            address: v.as_ptr() as u64,
            bytes: std::mem::size_of_val(v) as u64,
        };
        RegionHostV1 {
            regions: span(self.regions),
            half_valid: span(self.half_valid),
        }
    }

    /// Allocation-free diagnostic oracle, independent of GPU compaction.
    /// Returns selected indices in descending index order, padding with zero.
    /// Finite-domain baseline qualification is separate from this defined
    /// behavior on all F32 bit patterns. -inf is legal, including padding.
    pub fn select_row(self, scores: &[f32], indices: &mut [i32]) -> Result<u32, SageVsaFp4Error> {
        if scores.len() != self.shape.key_blocks() as usize || indices.len() != scores.len() {
            return Err(SageVsaFp4Error::Extent);
        }
        indices.fill(0);
        let mut count = 0;
        for key in (0..scores.len()).rev() {
            let selected = if key < self.prefix_blocks() as usize {
                true
            } else {
                let r = self
                    .regions
                    .chunks_exact(3)
                    .find(|r| key < r[1] as usize)
                    .ok_or(SageVsaFp4Error::Lists)?;
                let rank = (r[0] as usize..r[1] as usize)
                    .filter(|&other| precedes(scores[other], other, scores[key], key))
                    .count();
                rank < r[2] as usize
            };
            if selected {
                indices[count] = key as i32;
                count += 1;
            }
        }
        Ok(count as u32)
    }
}

fn precedes(a: f32, ai: usize, b: f32, bi: usize) -> bool {
    // Fixed Torch TopKTypeConfig<float>::convert: positive zero sorts above
    // negative zero and all NaNs sort above +inf, independent of NaN payload.
    // The exact implementation's index choice at cutoff ties still requires
    // qualification; this profile explicitly chooses the smaller index.
    let key = |v: f32| {
        let bits = v.to_bits();
        if bits & 0x7fffffff > 0x7f800000 {
            u32::MAX
        } else if bits & 0x80000000 != 0 {
            !bits
        } else {
            bits ^ 0x80000000
        }
    };
    let (a, b) = (key(a), key(b));
    a > b || (a == b && ai < bi)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RegionRequestV1 {
    pub attention: SageVsaFp4RequestV1,
    pub region_count: u32,
    pub selection_profile: u32,
}
impl RegionRequestV1 {
    pub fn validate(&self) -> Result<(), SageVsaFp4Error> {
        self.attention.validate()?;
        if self.selection_profile != REGION_SELECTION_PROFILE
            || self.region_count == 0
            || self.region_count > self.attention.query_blocks
            || self.attention.query_blocks >= self.attention.key_blocks
        {
            return Err(SageVsaFp4Error::UnsupportedGeometry);
        }
        self.score_bytes()?;
        Ok(())
    }
    pub fn score_bytes(&self) -> Result<u64, SageVsaFp4Error> {
        let r = self.attention;
        extent(&[
            r.batches.into(),
            r.heads.into(),
            r.key_blocks.into(),
            r.key_blocks.into(),
            4,
        ])
    }
    pub fn region_bytes(&self) -> u64 {
        u64::from(self.region_count) * 12
    }
}

/// `attention` retains the original checked kernel plan. Its three-upload
/// count describes the Host-list entry only. This entry instead performs two
/// geometry uploads, one memset and two kernels (selection then attention).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RegionPlanV1 {
    pub attention: SageVsaFp4PlanV1,
    pub region_count: u32,
    pub selection_profile: u32,
    pub selection_threads: u32,
    pub selection_grid_blocks: u32,
    pub upload_count: u32,
    pub memset_count: u32,
    pub scores_bytes: u64,
    pub regions_bytes: u64,
}
impl RegionPlanV1 {
    pub fn request(self) -> RegionRequestV1 {
        RegionRequestV1 {
            attention: self.attention.request,
            region_count: self.region_count,
            selection_profile: self.selection_profile,
        }
    }
    pub fn validate(&self) -> Result<(), SageVsaFp4Error> {
        self.attention.validate()?;
        let r = self.request();
        r.validate()?;
        let a = r.attention;
        if self.selection_threads != 128
            || self.upload_count != 2
            || self.memset_count != 1
            || u64::from(self.selection_grid_blocks)
                != u64::from(a.batches) * u64::from(a.heads) * u64::from(a.query_blocks)
            || self.scores_bytes != r.score_bytes()?
            || self.regions_bytes != r.region_bytes()
        {
            return Err(SageVsaFp4Error::Plan);
        }
        Ok(())
    }
    pub fn fingerprint_material(&self) -> [u8; 232] {
        let mut bytes = [0; 232];
        bytes[..192].copy_from_slice(&self.attention.fingerprint_material());
        for (dst, value) in bytes[192..216].chunks_exact_mut(4).zip([
            self.region_count,
            self.selection_profile,
            self.selection_threads,
            self.selection_grid_blocks,
            self.upload_count,
            self.memset_count,
        ]) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        bytes[216..224].copy_from_slice(&self.scores_bytes.to_le_bytes());
        bytes[224..232].copy_from_slice(&self.regions_bytes.to_le_bytes());
        bytes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RegionBindingsV1 {
    pub attention: SageVsaFp4BindingsV1,
    pub scores: SageVsaFp4SpanV1,
    pub regions: SageVsaFp4SpanV1,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RegionHostV1 {
    pub regions: SageVsaFp4SpanV1,
    pub half_valid: SageVsaFp4SpanV1,
}
const _: () = {
    assert!(size_of::<RegionRequestV1>() == 72);
    assert!(size_of::<RegionPlanV1>() == 232);
    assert!(size_of::<RegionBindingsV1>() == 224);
    assert!(size_of::<RegionHostV1>() == 32);
};

#[cfg(test)]
mod tests;
