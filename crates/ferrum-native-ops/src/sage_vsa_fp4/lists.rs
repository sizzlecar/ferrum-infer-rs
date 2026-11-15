use super::*;
use std::mem::size_of_val;

/// Device-independent logical shape of the checked CPU-list profile. Shape
/// validation grants neither native device support nor resource ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SageVsaFp4ListShape {
    batches: u32,
    heads: u32,
    query_blocks: u32,
    key_blocks: u32,
}
impl SageVsaFp4ListShape {
    pub fn new(
        batches: u32,
        heads: u32,
        query_blocks: u32,
        key_blocks: u32,
    ) -> Result<Self, SageVsaFp4Error> {
        if [batches, heads, query_blocks, key_blocks]
            .iter()
            .any(|&v| v == 0 || v > i32::MAX as u32)
            || query_blocks > key_blocks
            || extent(&[
                u64::from(batches),
                u64::from(heads),
                u64::from(key_blocks),
                128,
                128,
            ])? > i32::MAX as u64
        {
            return Err(SageVsaFp4Error::Extent);
        }
        Ok(Self {
            batches,
            heads,
            query_blocks,
            key_blocks,
        })
    }
    pub const fn batches(self) -> u32 {
        self.batches
    }
    pub const fn heads(self) -> u32 {
        self.heads
    }
    pub const fn query_blocks(self) -> u32 {
        self.query_blocks
    }
    pub const fn key_blocks(self) -> u32 {
        self.key_blocks
    }
    pub fn list_bytes(self) -> Result<[u64; 3], SageVsaFp4Error> {
        let rows = extent(&[
            self.batches.into(),
            self.heads.into(),
            self.query_blocks.into(),
        ])?;
        Ok([
            extent(&[rows, self.key_blocks.into(), 4])?,
            extent(&[rows, 4])?,
            extent(&[self.key_blocks.into(), 2, 4])?,
        ])
    }
}

/// Borrowed, allocation-free content proof. The borrow prevents mutation in
/// safe Rust. It does not prove pinning, admission, upload, or device readiness.
#[derive(Clone, Copy, Debug)]
pub struct CheckedCpuBlockLists<'a> {
    shape: SageVsaFp4ListShape,
    indices: &'a [i32],
    counts: &'a [i32],
    half_valid: &'a [i32],
}
impl<'a> CheckedCpuBlockLists<'a> {
    pub fn new(
        request: SageVsaFp4RequestV1,
        indices: &'a [i32],
        counts: &'a [i32],
        half_valid: &'a [i32],
    ) -> Result<Self, SageVsaFp4Error> {
        request.validate()?;
        Self::for_shape(
            SageVsaFp4ListShape::new(
                request.batches,
                request.heads,
                request.query_blocks,
                request.key_blocks,
            )?,
            indices,
            counts,
            half_valid,
        )
    }
    pub fn for_shape(
        shape: SageVsaFp4ListShape,
        indices: &'a [i32],
        counts: &'a [i32],
        half_valid: &'a [i32],
    ) -> Result<Self, SageVsaFp4Error> {
        let [index_bytes, count_bytes, half_bytes] = shape.list_bytes()?;
        if indices.len() as u64 != index_bytes / 4
            || counts.len() as u64 != count_bytes / 4
            || half_valid.len() as u64 != half_bytes / 4
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
        let keys = shape.key_blocks as usize;
        for (row, &count) in indices.chunks_exact(keys).zip(counts) {
            if count < 1 || count as usize > keys {
                return Err(SageVsaFp4Error::Lists);
            }
            let valid = &row[..count as usize];
            if valid.last() != Some(&0)
                || valid.iter().any(|&v| v < 0 || v as usize >= keys)
                || valid.windows(2).any(|pair| pair[0] <= pair[1])
            {
                return Err(SageVsaFp4Error::Lists);
            }
        }
        Ok(Self {
            shape,
            indices,
            counts,
            half_valid,
        })
    }
    pub fn shape(&self) -> SageVsaFp4ListShape {
        self.shape
    }
    pub fn indices(&self) -> &'a [i32] {
        self.indices
    }
    pub fn counts(&self) -> &'a [i32] {
        self.counts
    }
    pub fn half_valid(&self) -> &'a [i32] {
        self.half_valid
    }
    /// Caller must establish pinned Core-owned staging before passing these
    /// addresses to native Run. Ordinary Vec-backed proof is not submit-ready.
    pub fn host_addresses(&self) -> SageVsaFp4HostListsV1 {
        let span = |s: &[i32]| SageVsaFp4SpanV1 {
            address: s.as_ptr() as u64,
            bytes: size_of_val(s) as u64,
        };
        SageVsaFp4HostListsV1 {
            indices: span(self.indices),
            counts: span(self.counts),
            half_valid: span(self.half_valid),
        }
    }
}
