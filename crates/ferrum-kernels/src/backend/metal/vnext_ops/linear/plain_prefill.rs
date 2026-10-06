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
}

impl PlainLinearPlan {
    pub(super) fn for_launch(launch: LinearLaunch) -> Self {
        Self::split(launch).unwrap_or(Self::Single)
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
            // Eight F32 rows otherwise reread and decode Q6K weights eight
            // times. Two existing B4 kernels share each decode across four
            // rows. Keep unmeasured widths and partial groups on their prior
            // path; this is a shape/format policy, independent of model names.
            (LinearPhysicalFormat::Q6K, ElementType::F32, 8) => 4,
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
        }
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
