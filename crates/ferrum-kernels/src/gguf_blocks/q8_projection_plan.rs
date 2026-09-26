//! Numeric plan for the existing F16-input, F32-scale Q8 projection family.
//! It carries no device pointers, retained weights or execution permission.
use super::GgufBlockFormat;

pub(crate) const MAX_LEAF_ROWS: u64 = u16::MAX as u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Q8SumPolicy {
    Quantized,
    Input,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackLayout {
    pub scales_bytes: u64,
    pub sums_bytes: u64,
    pub words_offset: u64,
    pub total_bytes: u64,
}

impl PackLayout {
    pub fn new(rows: u64, columns: u64) -> Result<Self, String> {
        Self::with_policy(rows, columns, Q8SumPolicy::Quantized)
    }

    pub fn with_policy(rows: u64, columns: u64, policy: Q8SumPolicy) -> Result<Self, String> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) {
            return Err("Q8 activation packing requires rows and complete K256 blocks".into());
        }
        let values = rows
            .checked_mul(columns)
            .ok_or("Q8 activation extent overflows")?;
        let scales_bytes = values / 8;
        let sums_bytes = if policy == Q8SumPolicy::Input {
            scales_bytes
        } else {
            0
        };
        let words_offset = scales_bytes
            .checked_add(sums_bytes)
            .ok_or("Q8 metadata overflows")?;
        let total_bytes = values
            .checked_add(words_offset)
            .ok_or("Q8 workspace overflows")?;
        Ok(Self {
            scales_bytes,
            sums_bytes,
            words_offset,
            total_bytes,
        })
    }
}

/// An ephemeral projection of checked native matrix metadata, never a copy of
/// the matrix inventory or its backing storage. Dense F16 has no block format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixPart {
    pub format: Option<GgufBlockFormat>,
    pub outputs: u32,
    pub columns: u32,
    pub output_offset: u32,
    pub transformed: bool,
}

pub(crate) fn quantizes(format: Option<GgufBlockFormat>) -> bool {
    matches!(
        format,
        Some(GgufBlockFormat::Q4K | GgufBlockFormat::Q5K | GgufBlockFormat::Q6K)
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantizedKernel {
    Scalar,
    Tiled,
    Mma,
}

impl QuantizedKernel {
    pub fn tiles(self) -> (u32, u32) {
        match self {
            Self::Scalar => (1, 4),
            Self::Tiled => (8, 4),
            Self::Mma => (32, 16),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixPlan {
    whole_rows: u64,
    columns: u32,
    output_stride: u32,
    policy: Q8SumPolicy,
    projection_dispatches: u64,
    pack_bytes_per_row: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LeafPlan {
    pub rows: u32,
    pub policy: Q8SumPolicy,
    pub pack: Option<PackLayout>,
    pub quantized_kernel: Option<QuantizedKernel>,
    pub dispatches: u64,
}

impl MatrixPlan {
    /// Whole-invocation arithmetic is fixed before physical leaf selection.
    /// Parts are visited once; no allocation or inventory clone is performed.
    pub fn new(
        whole_rows: u64,
        columns: u32,
        output_stride: u32,
        policy: Q8SumPolicy,
        parts: impl IntoIterator<Item = MatrixPart>,
    ) -> Result<Self, String> {
        if whole_rows == 0 || columns == 0 || output_stride == 0 {
            return Err("Q8 matrix plan has an empty extent".into());
        }
        let mut projection_dispatches = 0_u64;
        let mut packed = false;
        for part in parts {
            if part.transformed
                || part.columns != columns
                || part.outputs == 0
                || part
                    .output_offset
                    .checked_add(part.outputs)
                    .is_none_or(|end| end > output_stride)
                || part.format.is_some_and(|format| {
                    !u64::from(columns).is_multiple_of(format.block_values() as u64)
                })
            {
                return Err("Q8 projection requires untransformed, matching matrix parts".into());
            }
            projection_dispatches = projection_dispatches
                .checked_add(1)
                .ok_or("Q8 matrix inventory overflows")?;
            packed |= quantizes(part.format);
        }
        if projection_dispatches == 0 {
            return Err("Q8 matrix plan has no parts".into());
        }
        let pack_bytes_per_row = if packed {
            PackLayout::with_policy(1, u64::from(columns), policy)?.total_bytes
        } else {
            0
        };
        pack_bytes_per_row
            .checked_mul(whole_rows)
            .ok_or("Q8 whole pack extent overflows")?;
        let plan = Self {
            whole_rows,
            columns,
            output_stride,
            policy,
            projection_dispatches,
            pack_bytes_per_row,
        };
        plan.dispatches(whole_rows)?;
        Ok(plan)
    }

    pub fn policy(self) -> Q8SumPolicy {
        self.policy
    }
    pub fn columns(self) -> u32 {
        self.columns
    }
    pub fn output_stride(self) -> u32 {
        self.output_stride
    }
    pub fn pack_bytes_per_row(self) -> u64 {
        self.pack_bytes_per_row
    }

    pub fn workspace_bytes(self, rows: u64) -> Result<u64, String> {
        self.check_rows(rows)?;
        self.pack_bytes_per_row
            .checked_mul(rows)
            .ok_or_else(|| "Q8 pack extent overflows".into())
    }

    /// Each MAX_LEAF_ROWS chunk repacks the corresponding current input rows.
    pub fn dispatches(self, rows: u64) -> Result<u64, String> {
        self.check_rows(rows)?;
        self.projection_dispatches
            .checked_add(u64::from(self.pack_bytes_per_row != 0))
            .and_then(|count| count.checked_mul(rows.div_ceil(MAX_LEAF_ROWS)))
            .ok_or_else(|| "Q8 projection dispatch count overflows".into())
    }

    pub fn leaf(self, rows: u64) -> Result<LeafPlan, String> {
        self.check_rows(rows)?;
        if rows > MAX_LEAF_ROWS {
            return Err("Q8 projection leaf exceeds the launch extent".into());
        }
        let pack = if self.pack_bytes_per_row != 0 {
            Some(PackLayout::with_policy(
                rows,
                u64::from(self.columns),
                self.policy,
            )?)
        } else {
            None
        };
        let quantized_kernel = pack.map(|_| {
            if rows >= 8 {
                QuantizedKernel::Mma
            } else if rows > 1 {
                QuantizedKernel::Tiled
            } else {
                QuantizedKernel::Scalar
            }
        });
        Ok(LeafPlan {
            rows: rows as u32,
            policy: self.policy,
            pack,
            quantized_kernel,
            dispatches: self.dispatches(rows)?,
        })
    }

    fn check_rows(self, rows: u64) -> Result<(), String> {
        if rows == 0 || rows > self.whole_rows {
            Err("Q8 projection rows exceed the whole invocation".into())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests;
