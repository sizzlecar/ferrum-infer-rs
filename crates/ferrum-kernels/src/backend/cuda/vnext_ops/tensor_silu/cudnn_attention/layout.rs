//! Pure checked scheduling and scratch layout; no native calls or allocations.

use super::*;

pub(super) const SCRATCH_ALIGNMENT: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Method {
    Causal,
    Empty,
    Sdpa { packed_rows: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Stage {
    pub first_query: u32,
    pub query_count: u32,
    pub first_batch: u32,
    pub batches: u32,
    pub method: Method,
}

#[derive(Debug)]
pub(super) struct Schedule {
    pub geometry: StructuredAttentionBf16Geometry,
    pub mask: attention::NativeMask,
    pub bytes: u64,
    pub stages: Vec<Stage>,
}

impl Schedule {
    pub fn new(geometry: StructuredAttentionBf16Geometry) -> Result<Self, String> {
        if geometry.head_dim() != 128 {
            return Err("cuDNN structured attention supports head_dim=128 only".into());
        }
        // cuDNN's GPU sequence-length tensors contain signed i32 values.
        if geometry.sequence_length() > i32::MAX as u64 {
            return Err("cuDNN attention sequence length exceeds i32".into());
        }
        let bytes = geometry
            .element_count()
            .checked_mul(2)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or("cuDNN attention tensor byte extent exceeds address space")?;
        let mask = attention::NativeMask::new(geometry.mask())?;
        let mut stages = Vec::new();
        let mut start = 0;
        for (&end, &bidirectional) in geometry
            .mask()
            .segment_ends()
            .iter()
            .zip(geometry.mask().segment_bidirectional())
        {
            let stage = Stage {
                first_query: start,
                query_count: end - start,
                first_batch: 0,
                batches: geometry.batch() as u32,
                method: Method::Causal,
            };
            if !bidirectional {
                stages.push(stage);
            } else {
                let mut lengths = [0u32; 16];
                for (batch, length) in lengths[..geometry.batch() as usize].iter_mut().enumerate() {
                    *length = valid_prefix(geometry.mask(), batch, end)?;
                }
                let lengths = &lengths[..geometry.batch() as usize];
                if lengths.iter().all(|&n| n != 0) {
                    // The normal batched path uses one cuDNN launch and exact
                    // per-batch lengths, even if padding differs by sample.
                    stages.push(Stage {
                        method: Method::Sdpa {
                            packed_rows: *lengths.iter().max().unwrap(),
                        },
                        ..stage
                    });
                } else if lengths.iter().all(|&n| n == 0) {
                    stages.push(Stage {
                        method: Method::Empty,
                        ..stage
                    });
                } else {
                    // Empty samples never enter cuDNN and never read Q/K/V.
                    // This avoids assuming vendor support for seq_len_kv=0.
                    for (batch, &length) in lengths.iter().enumerate() {
                        stages.push(Stage {
                            first_batch: batch as u32,
                            batches: 1,
                            method: if length == 0 {
                                Method::Empty
                            } else {
                                Method::Sdpa {
                                    packed_rows: length,
                                }
                            },
                            ..stage
                        });
                    }
                }
            }
            start = end;
        }
        Ok(Self {
            geometry,
            mask,
            bytes,
            stages,
        })
    }

    pub fn signature(&self, stage: Stage) -> Result<CudnnSdpaSignature, String> {
        let Method::Sdpa { packed_rows } = stage.method else {
            return Err("only a nonempty bidirectional stage has an SDPA signature".into());
        };
        let columns = self.geometry.columns();
        let source_stride = self
            .geometry
            .sequence_length()
            .checked_mul(columns)
            .ok_or("cuDNN attention source stride overflows")?;
        let packed_stride = u64::from(packed_rows)
            .checked_mul(columns)
            .ok_or("cuDNN attention packed stride overflows")?;
        let checked =
            |n| i64::try_from(n).map_err(|_| "cuDNN attention stride exceeds i64".to_owned());
        Ok(CudnnSdpaSignature {
            batch: i64::from(stage.batches),
            heads: checked(self.geometry.heads())?,
            query_length: i64::from(stage.query_count),
            key_value_length: i64::from(packed_rows),
            head_dim: 128,
            query_batch_stride: checked(source_stride)?,
            key_value_batch_stride: checked(packed_stride)?,
            output_batch_stride: checked(source_stride)?,
            scale_bits: self.geometry.effective_scale().to_bits(),
        })
    }
}

pub(super) fn valid_prefix(
    mask: &StructuredAttentionBf16Mask,
    batch: usize,
    end: u32,
) -> Result<u32, String> {
    let ends = mask.key_run_ends(batch).ok_or("missing key runs")?;
    let valid = mask.key_run_valid(batch).ok_or("missing key validity")?;
    let mut begin = 0;
    let mut count = 0u32;
    for (&run_end, &included) in ends.iter().zip(valid) {
        let clipped = run_end.min(end);
        if included {
            count = count
                .checked_add(clipped - begin)
                .ok_or("valid key count overflows")?;
        }
        if clipped == end {
            break;
        }
        begin = clipped;
    }
    Ok(count)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ScratchLayout {
    pub value_offset: u64,
    pub query_lengths_offset: u64,
    pub key_lengths_offset: u64,
    pub workspace_offset: u64,
    pub workspace_bytes: u64,
    pub bytes: u64,
}

impl ScratchLayout {
    pub fn new(
        batches: u32,
        packed_rows: u32,
        columns: u64,
        workspace_bytes: u64,
    ) -> Result<Self, String> {
        if batches == 0 || batches > 16 || packed_rows == 0 || columns == 0 {
            return Err("invalid cuDNN attention packing extent".into());
        }
        let payload = u64::from(batches)
            .checked_mul(u64::from(packed_rows))
            .and_then(|n| n.checked_mul(columns))
            .and_then(|n| n.checked_mul(2))
            .ok_or("cuDNN attention packing bytes overflow")?;
        let align = |n: u64| {
            n.checked_add(SCRATCH_ALIGNMENT - 1)
                .map(|n| n / SCRATCH_ALIGNMENT * SCRATCH_ALIGNMENT)
                .ok_or_else(|| "cuDNN attention scratch alignment overflows".to_owned())
        };
        let value_offset = align(payload)?;
        let query_lengths_offset = align(
            value_offset
                .checked_add(payload)
                .ok_or("cuDNN attention scratch overflow")?,
        )?;
        let lengths_bytes = u64::from(batches) * 4;
        let key_lengths_offset = align(
            query_lengths_offset
                .checked_add(lengths_bytes)
                .ok_or("cuDNN attention length scratch overflows")?,
        )?;
        let workspace_offset = align(
            key_lengths_offset
                .checked_add(lengths_bytes)
                .ok_or("cuDNN attention length scratch overflows")?,
        )?;
        let bytes = workspace_offset
            .checked_add(workspace_bytes)
            .filter(|&n| n <= isize::MAX as u64)
            .ok_or("cuDNN attention scratch exceeds address space")?;
        Ok(Self {
            value_offset,
            query_lengths_offset,
            key_lengths_offset,
            workspace_offset,
            workspace_bytes,
            bytes,
        })
    }
}
