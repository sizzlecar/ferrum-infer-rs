//! Explicit charged Host payload; a digest is never a replacement for ownership.
use ferrum_native_ops::sage_vsa_fp4::{CheckedCpuBlockLists, SageVsaFp4ListShape};
use ferrum_runtime::{HostMemoryClass, ManagedHostBytes};
use sha2::{Digest, Sha256};

pub const CHECKED_HOST_PATTERN_SOURCE: &str = "sparse-fp4.checked-immutable-host-pattern.v1";

/// No constructor allocates payload storage. The caller supplies immutable
/// ManagedHost allocations, with page padding already charged to its existing
/// Host account. Clones and slices retain those original physical charges.
/// CUDA page registration is a separate, bounded runtime preparation step.
pub struct CheckedHostVsaPattern {
    shape: SageVsaFp4ListShape,
    // Retain the caller's charged immutable owners even in builds without CUDA.
    #[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
    payloads: [ManagedHostBytes; 3],
    digest: String,
    selected_tiles: u64,
}
impl CheckedHostVsaPattern {
    pub fn new(
        shape: SageVsaFp4ListShape,
        indices: ManagedHostBytes,
        counts: ManagedHostBytes,
        half_valid: ManagedHostBytes,
    ) -> Result<Self, String> {
        let payloads = [indices, counts, half_valid];
        let expected = shape.list_bytes().map_err(|e| e.to_string())?;
        for (payload, bytes) in payloads.iter().zip(expected) {
            if payload.class() != HostMemoryClass::TransferStaging || payload.len() as u64 != bytes
            {
                return Err(
                    "VSA Host pattern requires exact charged TransferStaging payloads".into(),
                );
            }
        }
        let indices = integers(&payloads[0])?;
        let counts = integers(&payloads[1])?;
        let half = integers(&payloads[2])?;
        CheckedCpuBlockLists::for_shape(shape, indices, counts, half).map_err(|e| e.to_string())?;
        let selected_tiles = counts
            .iter()
            .try_fold(0u64, |n, &c| n.checked_add(c as u64))
            .ok_or("VSA selected tile count overflow")?;
        let mut hash = Sha256::new();
        hash.update(CHECKED_HOST_PATTERN_SOURCE.as_bytes());
        for value in [
            shape.batches(),
            shape.heads(),
            shape.query_blocks(),
            shape.key_blocks(),
        ] {
            hash.update(value.to_le_bytes());
        }
        for payload in &payloads {
            hash.update((payload.len() as u64).to_le_bytes());
            hash.update(payload.bytes());
        }
        Ok(Self {
            shape,
            payloads,
            digest: format!("{:x}", hash.finalize()),
            selected_tiles,
        })
    }
    pub const fn shape(&self) -> SageVsaFp4ListShape {
        self.shape
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub const fn selected_tiles(&self) -> u64 {
        self.selected_tiles
    }
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn payloads(&self) -> &[ManagedHostBytes; 3] {
        &self.payloads
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn checked_lists(&self) -> Result<CheckedCpuBlockLists<'_>, String> {
        CheckedCpuBlockLists::for_shape(
            self.shape,
            integers(&self.payloads[0])?,
            integers(&self.payloads[1])?,
            integers(&self.payloads[2])?,
        )
        .map_err(|e| e.to_string())
    }
}
pub(super) fn integers(bytes: &ManagedHostBytes) -> Result<&[i32], String> {
    let data = bytes.bytes();
    if !cfg!(target_endian = "little")
        || data.as_ptr() as usize % std::mem::align_of::<i32>() != 0
        || data.len() % 4 != 0
    {
        return Err("VSA Host lists require aligned little-endian I32 payloads".into());
    }
    // SAFETY: checked alignment/exact multiple; ManagedHostBytes owns stable,
    // initialized immutable bytes for this borrow. Every i32 bit pattern is valid.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), data.len() / 4) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_runtime::{HostMemoryAccount, HostMemoryClassLimits, HostMemoryLimits};
    fn account(alignment: usize, total: usize) -> HostMemoryAccount {
        HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                total,
                HostMemoryClassLimits {
                    input_preprocess_bytes: total,
                    transfer_staging_bytes: total,
                    output_bytes: total,
                    weight_backing_bytes: total,
                },
            )
            .unwrap(),
            alignment,
        )
        .unwrap()
    }
    fn payload(
        host: &HostMemoryAccount,
        values: &[i32],
        class: HostMemoryClass,
    ) -> ManagedHostBytes {
        let mut bytes = host.allocate_zeroed(class, values.len() * 4).unwrap();
        for (dst, value) in bytes.as_mut().chunks_exact_mut(4).zip(values) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
        bytes.freeze()
    }
    fn pattern(
        host: &HostMemoryAccount,
        indices: &[i32],
        counts: &[i32],
        half: &[i32],
    ) -> Result<CheckedHostVsaPattern, String> {
        CheckedHostVsaPattern::new(
            SageVsaFp4ListShape::new(1, 1, 1, 2).unwrap(),
            payload(host, indices, HostMemoryClass::TransferStaging),
            payload(host, counts, HostMemoryClass::TransferStaging),
            payload(host, half, HostMemoryClass::TransferStaging),
        )
    }
    #[test]
    fn sage_pattern_retains_original_immutable_allocations_and_physical_padding() {
        let alignment = 4096;
        let physical = 3 * (2 * alignment - 1);
        let host = account(alignment, physical);
        let indices = payload(&host, &[1, 0], HostMemoryClass::TransferStaging);
        let alias = indices.clone();
        let address = indices.bytes().as_ptr();
        let pattern = CheckedHostVsaPattern::new(
            SageVsaFp4ListShape::new(1, 1, 1, 2).unwrap(),
            indices,
            payload(&host, &[2], HostMemoryClass::TransferStaging),
            payload(&host, &[64, 64, 64, 64], HostMemoryClass::TransferStaging),
        )
        .unwrap();
        assert_eq!(pattern.payloads()[0].bytes().as_ptr(), address);
        assert_eq!(pattern.selected_tiles(), 2);
        assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
        assert!(host
            .allocate_zeroed(HostMemoryClass::TransferStaging, 1)
            .is_err());
        let alias = alias.try_thaw().unwrap_err(); // prepared payload cannot be mutated through a caller alias
        drop(pattern);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 2 * alignment - 1);
        let writable = alias.try_thaw().unwrap();
        assert_eq!(writable.as_ref().as_ptr(), address);
        drop(writable);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
    #[test]
    fn sage_pattern_digest_binds_contents_shape_and_explicit_profile() {
        let host = account(16, 4096);
        let full = pattern(&host, &[1, 0], &[2], &[64, 64, 64, 64]).unwrap();
        let again = pattern(&host, &[1, 0], &[2], &[64, 64, 64, 64]).unwrap();
        assert_eq!(full.digest(), again.digest());
        let sparse = pattern(&host, &[0, 0], &[1], &[64, 64, 64, 64]).unwrap();
        let partial = pattern(&host, &[1, 0], &[2], &[64, 64, 17, 0]).unwrap();
        assert_ne!(full.digest(), sparse.digest());
        assert_ne!(full.digest(), partial.digest());
        assert_eq!(sparse.selected_tiles(), 1);
        assert_eq!(full.digest().len(), 64);
    }
    #[test]
    fn sage_pattern_rejects_wrong_owner_class_content_alignment_and_full_table_extent() {
        let host = account(16, 4096);
        for (indices, counts, half) in [
            (&[1, 0, 1, 0][..], &[2][..], &[64, 64, 64, 64][..]),
            (&[0, 1][..], &[2][..], &[64, 64, 64, 64][..]),
            (&[1, 0][..], &[0][..], &[64, 64, 64, 64][..]),
            (&[1, 0][..], &[2][..], &[64, 64, 63, 1][..]),
        ] {
            assert!(pattern(&host, indices, counts, half).is_err());
        }
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        let shape = SageVsaFp4ListShape::new(1, 1, 1, 2).unwrap();
        assert!(CheckedHostVsaPattern::new(
            shape,
            payload(&host, &[1, 0], HostMemoryClass::WeightBacking),
            payload(&host, &[2], HostMemoryClass::TransferStaging),
            payload(&host, &[64, 64, 64, 64], HostMemoryClass::TransferStaging)
        )
        .is_err());
        let backing = payload(&host, &[0, 0, 0], HostMemoryClass::TransferStaging);
        let unaligned = backing.slice(1..9).unwrap();
        assert!(CheckedHostVsaPattern::new(
            shape,
            unaligned,
            payload(&host, &[1], HostMemoryClass::TransferStaging),
            payload(&host, &[64, 64, 64, 64], HostMemoryClass::TransferStaging)
        )
        .is_err());
        drop(backing);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    }
}
