//! Cold qualification of one complete MMVQ launch chain. This changes
//! scheduling only; numerical/weight-decode selection remains authoritative.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MmvqLaunchKind {
    #[default]
    Ordered,
    Programmatic,
}
impl MmvqLaunchKind {
    pub const fn as_runtime_value(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Programmatic => "programmatic",
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
#[cfg(any(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl), test))]
pub(super) enum DotKind {
    Ordinary = 0,
    Q4Shared = 1,
    Q5Shared = 2,
}
#[cfg(any(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl), test))]
pub(super) fn dot_kind(plan: &PreparedUpstreamLinear) -> DotKind {
    if plan.q4_weight_decode != ferrum_types::Q4MmvqWeightDecode::Baseline {
        DotKind::Q4Shared
    } else if plan.q5_weight_decode != ferrum_types::Q5MmvqWeightDecode::Baseline {
        DotKind::Q5Shared
    } else {
        DotKind::Ordinary
    }
}

fn eligible(plan: &PreparedUpstreamLinear) -> bool {
    #[cfg(any(feature = "cuda-upstream-linear", test))]
    {
        let p = &plan.raw;
        plan.family == Family::Original
            && plan.arithmetic == Arithmetic::MarkerV2
            && p.algorithm == Algorithm::Mmvq as u32
            && p.request.layout == Layout::Columns as u32
            && matches!(p.request.rows, 1 | 4 | 8)
            && p.ncols == p.request.rows
            && p.channels == 1
            && p.fixup_bytes == 0
            && matches!(p.request.format, v if v == Format::Q4K as u32
                || v == Format::Q5K as u32 || v == Format::Iq4Xs as u32)
    }
    #[cfg(not(any(feature = "cuda-upstream-linear", test)))]
    {
        let _ = plan;
        false
    }
}

pub(super) fn qualify(plan: &PreparedUpstreamLinear) -> Result<MmvqLaunchKind, Error> {
    if !eligible(plan) {
        return Ok(MmvqLaunchKind::Ordered);
    }
    #[cfg(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl))]
    {
        let mut supported = 0;
        Error::from_status(unsafe {
            ffi::ferrum_upstream_mmvq_pdl_support_v1(
                &plan.raw,
                dot_kind(plan) as u32,
                &mut supported,
            )
        })?;
        Ok(if supported == 0 {
            MmvqLaunchKind::Ordered
        } else {
            MmvqLaunchKind::Programmatic
        })
    }
    #[cfg(not(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_pdl)))]
    {
        Ok(MmvqLaunchKind::Ordered)
    }
}

fn boundary_eligible(plan: &PreparedUpstreamLinear) -> bool {
    eligible(plan)
        && plan.raw.request.rows == 1
        && plan.q4_weight_decode == ferrum_types::Q4MmvqWeightDecode::Baseline
        && plan.q5_weight_decode == ferrum_types::Q5MmvqWeightDecode::Baseline
}

pub(super) fn qualify_boundary(plan: &PreparedUpstreamLinear) -> Result<MmvqBoundaryKind, Error> {
    if !boundary_eligible(plan) {
        return Ok(MmvqBoundaryKind::F32Scratch);
    }
    #[cfg(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_m1_f16_pdl))]
    {
        let mut supported = 0;
        Error::from_status(unsafe {
            ffi::ferrum_upstream_mmvq_m1_f16_pdl_support_v1(&plan.raw, &mut supported)
        })?;
        Ok(if supported == 0 {
            MmvqBoundaryKind::F32Scratch
        } else {
            MmvqBoundaryKind::DirectF16
        })
    }
    #[cfg(not(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_m1_f16_pdl)))]
    {
        Ok(MmvqBoundaryKind::F32Scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prepared(rows: u32, format: Format) -> PreparedUpstreamLinear {
        let mut raw = UpstreamLinearPlanV1::default();
        raw.algorithm = Algorithm::Mmvq as u32;
        raw.request.rows = rows;
        raw.request.format = format as u32;
        raw.request.layout = Layout::Columns as u32;
        raw.ncols = rows;
        raw.channels = 1;
        PreparedUpstreamLinear {
            raw,
            arithmetic: Arithmetic::MarkerV2,
            family: Family::Original,
            small_mmq_pack: SmallMmqPack::LegacyF32,
            small_mmvq_pack: SmallMmvqPack::F32Memset,
            q4_weight_decode: ferrum_types::Q4MmvqWeightDecode::Baseline,
            q5_weight_decode: ferrum_types::Q5MmvqWeightDecode::Baseline,
            mmvq_launch: MmvqLaunchKind::Ordered,
            mmvq_boundary: MmvqBoundaryKind::F32Scratch,
        }
    }
    #[test]
    fn mmvq_pdl_eligibility_preserves_original_numeric_family_and_actual_columns() {
        for format in [Format::Q4K, Format::Q5K, Format::Iq4Xs] {
            for rows in [1, 4, 8] {
                assert!(eligible(&prepared(rows, format)));
            }
            for rows in [0, 2, 3, 5, 6, 7, 9, 32] {
                assert!(!eligible(&prepared(rows, format)));
            }
        }
        let original = prepared(8, Format::Q4K);
        for changed in 0..7 {
            let mut p = original.clone();
            match changed {
                0 => p.family = Family::Extra,
                1 => p.arithmetic = Arithmetic::V1,
                2 => p.raw.algorithm = Algorithm::Mmq as u32,
                3 => p.raw.request.layout = Layout::Channels as u32,
                4 => p.raw.ncols = 1,
                5 => p.raw.channels = 2,
                _ => p.raw.fixup_bytes = 4,
            }
            assert!(!eligible(&p));
        }
    }
    #[test]
    fn mmvq_pdl_dot_kind_preserves_selected_weight_decode_and_late_changes_reset_launch() {
        let mut p = prepared(8, Format::Q4K);
        assert!(matches!(dot_kind(&p), DotKind::Ordinary));
        p.q4_weight_decode = ferrum_types::Q4MmvqWeightDecode::SharedM8;
        assert!(matches!(dot_kind(&p), DotKind::Q4Shared));
        p.mmvq_launch = MmvqLaunchKind::Programmatic;
        let p = p
            .with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::Baseline)
            .unwrap();
        assert_eq!(p.mmvq_launch_kind(), MmvqLaunchKind::Ordered);
        assert!(matches!(dot_kind(&p), DotKind::Ordinary));
        let mut p = prepared(8, Format::Q5K);
        p.q5_weight_decode = ferrum_types::Q5MmvqWeightDecode::SharedM8;
        assert!(matches!(dot_kind(&p), DotKind::Q5Shared));
        p.mmvq_launch = MmvqLaunchKind::Programmatic;
        assert_eq!(
            p.with_q5_mmvq_weight_decode(ferrum_types::Q5MmvqWeightDecode::Baseline)
                .unwrap()
                .mmvq_launch_kind(),
            MmvqLaunchKind::Ordered
        );
    }
    #[test]
    fn mmvq_pdl_work_accounting_keeps_all_three_pack_kernels_and_no_transfers() {
        let mut p = prepared(1, Format::Q4K);
        p.mmvq_launch = MmvqLaunchKind::Programmatic;
        assert_eq!(p.pack_work_counts(), (3, 0));
        assert_ne!(
            MmvqLaunchKind::Ordered.as_runtime_value(),
            MmvqLaunchKind::Programmatic.as_runtime_value()
        );
    }
    #[test]
    fn mmvq_f16_boundary_eligibility_and_accounting_preserve_fallbacks() {
        for format in [Format::Q4K, Format::Q5K, Format::Iq4Xs] {
            assert!(boundary_eligible(&prepared(1, format)));
            for rows in [2, 4, 8, 32] {
                let p = prepared(rows, format)
                    .with_automatic_mmvq_boundary()
                    .unwrap();
                assert_eq!(p.mmvq_boundary_kind(), MmvqBoundaryKind::F32Scratch);
                assert_eq!(p.output_work_count(), 2);
            }
        }
        for change in 0..8 {
            let mut p = prepared(1, Format::Q4K);
            match change {
                0 => p.family = Family::Extra,
                1 => p.arithmetic = Arithmetic::V1,
                2 => p.raw.algorithm = Algorithm::Mmq as u32,
                3 => p.raw.request.layout = Layout::Channels as u32,
                4 => p.raw.ncols = 2,
                5 => p.raw.channels = 2,
                6 => p.raw.fixup_bytes = 4,
                _ => p.q4_weight_decode = ferrum_types::Q4MmvqWeightDecode::SharedM8,
            }
            assert!(!boundary_eligible(&p));
            assert_eq!(
                p.with_automatic_mmvq_boundary()
                    .unwrap()
                    .mmvq_boundary_kind(),
                MmvqBoundaryKind::F32Scratch
            );
        }
        let mut p = prepared(1, Format::Q4K);
        p.mmvq_launch = MmvqLaunchKind::Programmatic;
        p.mmvq_boundary = MmvqBoundaryKind::DirectF16;
        assert_eq!(p.pack_work_counts(), (2, 0));
        assert_eq!(p.output_work_count(), 1);
        assert_ne!(
            MmvqBoundaryKind::DirectF16.as_runtime_value(),
            MmvqBoundaryKind::F32Scratch.as_runtime_value()
        );
        assert_eq!(
            p.with_q4_mmvq_weight_decode(ferrum_types::Q4MmvqWeightDecode::Baseline)
                .unwrap()
                .mmvq_boundary_kind(),
            MmvqBoundaryKind::F32Scratch
        );
    }

    #[test]
    #[cfg(not(all(feature = "cuda-upstream-linear", ferrum_native_mmvq_m1_f16_pdl)))]
    fn mmvq_f16_boundary_old_archive_retains_f32_scratch() {
        let p = prepared(1, Format::Q4K)
            .with_automatic_mmvq_boundary()
            .unwrap();
        assert_eq!(p.mmvq_boundary_kind(), MmvqBoundaryKind::F32Scratch);
        assert_eq!(p.mmvq_launch_kind(), MmvqLaunchKind::Ordered);
    }
}
