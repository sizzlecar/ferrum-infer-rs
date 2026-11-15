//! A separate geometry transport; selection and attention keep their exact
//! original plans. Packets are CUDA kernel arguments, never pageable DMA.
use super::*;

pub const REGION_PACKET_BYTES: u64 = 512 * 4;
pub const REGION_KERNEL_EXPORTS: [&str; 3] = [
    "ferrum_sage_vsa_fp4_region_kernel_plan_v1",
    "ferrum_sage_vsa_fp4_region_kernel_validate_v1",
    "ferrum_sage_vsa_fp4_region_kernel_run_v1",
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct RegionKernelPlanV1 {
    pub geometry: RegionPlanV1,
    pub upload_kernel_count: u32,
    pub reserved: u32,
}
impl RegionKernelPlanV1 {
    pub fn from_geometry(geometry: RegionPlanV1) -> Result<Self, SageVsaFp4Error> {
        geometry.validate()?;
        let count = geometry
            .regions_bytes
            .div_ceil(REGION_PACKET_BYTES)
            .checked_add(
                geometry
                    .attention
                    .bytes
                    .half_valid
                    .div_ceil(REGION_PACKET_BYTES),
            )
            .and_then(|n| u32::try_from(n).ok())
            .ok_or(SageVsaFp4Error::Extent)?;
        Ok(Self {
            geometry,
            upload_kernel_count: count,
            reserved: 0,
        })
    }
    pub fn validate(&self) -> Result<(), SageVsaFp4Error> {
        if *self != Self::from_geometry(self.geometry)? {
            return Err(SageVsaFp4Error::Plan);
        }
        Ok(())
    }
    /// Exact native calls, including every bounded copy kernel. The nested
    /// geometry plan still describes the legacy transport's two logical tables.
    pub fn dispatch_counts(&self) -> (u64, u64) {
        (
            u64::from(self.upload_kernel_count) + 2,
            u64::from(self.geometry.memset_count),
        )
    }
    pub fn fingerprint_material(&self) -> [u8; 240] {
        let mut bytes = [0; 240];
        bytes[..232].copy_from_slice(&self.geometry.fingerprint_material());
        bytes[232..236].copy_from_slice(&self.upload_kernel_count.to_le_bytes());
        bytes[236..240].copy_from_slice(&self.reserved.to_le_bytes());
        bytes
    }
}
const _: () = assert!(size_of::<RegionKernelPlanV1>() == 240);

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(queries: u32, keys: u32, regions: u32) -> RegionPlanV1 {
        let request = SageVsaFp4RequestV1::new(
            1,
            1,
            queries,
            keys,
            0.125,
            SageVsaFp4Device {
                ordinal: 0,
                architecture: 1200,
                multiprocessors: 4,
                maximum_dynamic_shared_bytes: 131072,
            },
        )
        .unwrap();
        let attention = SageVsaFp4PlanV1 {
            request,
            abi: 1,
            size: 192,
            kernel_id: 1,
            shared_bytes: 4096,
            threads: 384,
            grid_blocks: 4,
            upload_count: 3,
            memset_count: 1,
            bytes: request.buffer_sizes().unwrap(),
        };
        let r = RegionRequestV1 {
            attention: request,
            region_count: regions,
            selection_profile: REGION_SELECTION_PROFILE,
        };
        RegionPlanV1 {
            attention,
            region_count: regions,
            selection_profile: REGION_SELECTION_PROFILE,
            selection_threads: 128,
            selection_grid_blocks: queries,
            upload_count: 2,
            memset_count: 1,
            scores_bytes: r.score_bytes().unwrap(),
            regions_bytes: r.region_bytes(),
        }
    }

    #[test]
    fn packet_boundaries_preserve_full_geometry_and_reject_underdeclared_work() {
        // Half-valid table straddles 2048 bytes; the last case also crosses
        // that boundary in the independent region table (171 triples).
        for (q, k, regions, uploads) in [
            (1, 255, 1, 2),
            (1, 256, 1, 2),
            (1, 257, 1, 3),
            (200, 257, 171, 4),
        ] {
            let geometry = geometry(q, k, regions);
            let plan = RegionKernelPlanV1::from_geometry(geometry).unwrap();
            plan.validate().unwrap();
            assert_eq!(plan.geometry, geometry);
            assert_eq!(plan.upload_kernel_count, uploads);
            assert_eq!(plan.dispatch_counts(), (u64::from(uploads) + 2, 1));
            let mut short = plan;
            short.upload_kernel_count -= 1;
            assert_eq!(short.validate(), Err(SageVsaFp4Error::Plan));
            let mut changed = plan;
            changed.reserved = 1;
            assert_eq!(changed.validate(), Err(SageVsaFp4Error::Plan));
            assert_ne!(plan.fingerprint_material(), short.fingerprint_material());
        }
    }
}
