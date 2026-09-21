//! Prepared physical row partitions for plain quantized projections.

use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) enum PlainLinearPlan {
    Single,
    SplitRows {
        prefix_rows: u32,
        tail_input_offset_bytes: u64,
        tail_output_offset_bytes: u64,
    },
    FourRowGroups {
        groups: u32,
        input_group_bytes: u64,
        output_group_bytes: u64,
    },
}

impl PlainLinearPlan {
    pub(super) fn for_launch(launch: LinearLaunch) -> Self {
        Self::four_row_groups(launch)
            .or_else(|| Self::split(launch))
            .unwrap_or(Self::Single)
    }

    fn four_row_groups(launch: LinearLaunch) -> Option<Self> {
        if launch.transform.is_some()
            || launch.format != LinearPhysicalFormat::Q6K
            || launch.activation_type != ElementType::F32
            || !(8..=32).contains(&launch.params.rows)
            || launch.params.out_features < SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES
        {
            return None;
        }
        // Reuse the measured B4 reduction at decode cohort widths. Leave
        // larger prefill batches on their existing route instead of growing
        // an unbounded sequence of small physical dispatches.
        let groups = launch.params.rows.div_ceil(4);
        let input_group_bytes = u64::from(launch.params.in_features) * 4 * 4;
        let output_group_bytes = u64::from(launch.params.output_stride) * 4 * 4;
        launch
            .input_offset_bytes
            .checked_add(u64::from(groups - 1).checked_mul(input_group_bytes)?)?;
        launch
            .output_offset_bytes
            .checked_add(u64::from(groups - 1).checked_mul(output_group_bytes)?)?;
        Some(Self::FourRowGroups {
            groups,
            input_group_bytes,
            output_group_bytes,
        })
    }

    fn split(launch: LinearLaunch) -> Option<Self> {
        if launch.transform.is_some()
            || launch.params.out_features < SHARED_WEIGHT_GEMV_MIN_OUTPUT_FEATURES
        {
            return None;
        }
        let prefix_rows = match (launch.format, launch.activation_type, launch.params.rows) {
            // A one-row GEMM tail otherwise computes a complete 32-row tile.
            (LinearPhysicalFormat::Q4K, ElementType::F16, rows) if rows > 32 && rows % 32 == 1 => {
                rows - 1
            }
            // C8 otherwise fills only a quarter of the 32-row MMA tile. Reuse
            // the existing shared-weight B4 kernels once K fills all four
            // 256-value block lanes in the Q4K reduction. Short K keeps its
            // established route, as do narrow N and unmeasured batch widths.
            (LinearPhysicalFormat::Q4K | LinearPhysicalFormat::Q6K, ElementType::F16, 8)
                if launch.params.in_features >= 1024 =>
            {
                4
            }
            // Q5K's expanding projection benefits from the existing tiled
            // GEMM; splitting it regresses paired measurements. Use B4 groups
            // only for contracting projections, retaining the tiled route for
            // equal or wider outputs until that range has measured benefit.
            (LinearPhysicalFormat::Q5K, ElementType::F16, 8)
                if launch.params.in_features >= 1024
                    && launch.params.out_features < launch.params.in_features =>
            {
                4
            }
            _ => return None,
        };
        let input_bytes = u64::from(prefix_rows)
            .checked_mul(u64::from(launch.params.in_features))?
            .checked_mul(launch.activation_type.size_bytes())?;
        let output_bytes = u64::from(prefix_rows)
            .checked_mul(u64::from(launch.params.output_stride))?
            .checked_mul(launch.activation_type.size_bytes())?;
        Some(Self::SplitRows {
            prefix_rows,
            tail_input_offset_bytes: launch.input_offset_bytes.checked_add(input_bytes)?,
            tail_output_offset_bytes: launch.output_offset_bytes.checked_add(output_bytes)?,
        })
    }

    pub(super) fn dispatch_count(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::SplitRows { .. } => 2,
            Self::FourRowGroups { groups, .. } => u64::from(groups),
        }
    }

    pub(super) fn grouped_parts(
        self,
        launch: LinearLaunch,
    ) -> Option<impl Iterator<Item = LinearLaunch>> {
        let Self::FourRowGroups {
            groups,
            input_group_bytes,
            output_group_bytes,
        } = self
        else {
            return None;
        };
        Some((0..groups).map(move |group| {
            let mut part = launch;
            part.params.rows = (launch.params.rows - group * 4).min(4);
            // The plan checked the final offset; every preceding group is
            // within the same validated, retained full-row regions. A tail
            // selects the existing one-, two- or three-row pipeline.
            part.input_offset_bytes += u64::from(group) * input_group_bytes;
            part.output_offset_bytes += u64::from(group) * output_group_bytes;
            part.plain_plan = Self::Single;
            part
        }))
    }

    pub(super) fn parts(self, launch: LinearLaunch) -> Option<[LinearLaunch; 2]> {
        let Self::SplitRows {
            prefix_rows,
            tail_input_offset_bytes,
            tail_output_offset_bytes,
        } = self
        else {
            return None;
        };
        // Normal launch validation covers the original complete rows. These
        // two disjoint row spans use the same retained regions and weight part,
        // so no additional scratch or authority is introduced by the split.
        let mut head = launch;
        head.params.rows = prefix_rows;
        head.plain_plan = Self::Single;
        let mut tail = launch;
        tail.params.rows -= prefix_rows;
        tail.input_offset_bytes = tail_input_offset_bytes;
        tail.output_offset_bytes = tail_output_offset_bytes;
        tail.plain_plan = Self::Single;
        Some([head, tail])
    }
}
