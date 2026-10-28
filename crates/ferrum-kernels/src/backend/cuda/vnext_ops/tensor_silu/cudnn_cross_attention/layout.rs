//! Checked rectangular scheduling; payload packing uses the existing kernel.
use super::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Method {
    Empty,
    Sdpa {
        packed_rows: u32,
        /// Every original KV row is visible. Its contiguous BSHD storage is
        /// identical to the packed layout, so only sequence lengths are filled.
        direct_kv: bool,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Stage {
    pub first_query: u32,
    pub query_count: u32,
    pub first_batch: u32,
    pub batches: u32,
    pub method: Method,
}
pub(super) struct Schedule {
    pub geometry: CrossAttentionBf16Geometry,
    pub masks: Vec<attention::NativeMask>,
    pub bytes: u64,
    pub stages: Vec<Stage>,
}
impl Schedule {
    pub fn new(geometry: CrossAttentionBf16Geometry) -> Result<Self, String> {
        if !matches!(geometry.head_dim(), 72 | 128)
            || geometry.query_length() > i32::MAX as u64
            || geometry.key_value_length() > i32::MAX as u64
        {
            return Err(
                "cuDNN cross attention requires D72/128 and positive i32 query/KV lengths".into(),
            );
        }
        let batch = geometry.batch() as usize;
        let mut stages = Vec::new();
        let mut masks = Vec::new();
        for first in (0..batch).step_by(attention::MASK_BATCH_TILE) {
            let count = attention::MASK_BATCH_TILE.min(batch - first);
            masks.push(attention::NativeMask::cross_range(&geometry, first, count)?);
            let base = Stage {
                first_query: 0,
                query_count: geometry.query_length() as u32,
                first_batch: first as u32,
                batches: count as u32,
                method: Method::Empty,
            };
            let lengths = (first..first + count)
                .map(|b| {
                    geometry
                        .valid_keys(b)
                        .ok_or("missing cross batch key count")
                })
                .collect::<Result<Vec<_>, _>>()?;
            if lengths.iter().all(|&n| n != 0) {
                stages.push(Stage {
                    method: Method::Sdpa {
                        packed_rows: *lengths.iter().max().unwrap(),
                        direct_kv: lengths
                            .iter()
                            .all(|&n| u64::from(n) == geometry.key_value_length()),
                    },
                    ..base
                });
            } else if lengths.iter().all(|&n| n == 0) {
                stages.push(base);
            } else {
                for (b, n) in lengths.into_iter().enumerate() {
                    stages.push(Stage {
                        first_batch: (first + b) as u32,
                        batches: 1,
                        method: if n == 0 {
                            Method::Empty
                        } else {
                            Method::Sdpa {
                                packed_rows: n,
                                direct_kv: false,
                            }
                        },
                        ..base
                    });
                }
            }
        }
        let result = Self {
            bytes: geometry.query_bytes(),
            geometry,
            masks,
            stages,
        };
        for &s in &result.stages {
            if let Method::Sdpa { .. } = s.method {
                result.signature(s)?.validate()?;
            }
        }
        Ok(result)
    }
    pub fn signature(&self, stage: Stage) -> Result<CudnnSdpaSignature, String> {
        let Method::Sdpa { packed_rows, .. } = stage.method else {
            return Err("empty cross stage has no vendor signature".into());
        };
        let g = &self.geometry;
        let checked = |n| i64::try_from(n).map_err(|_| "cross stride exceeds i64".to_owned());
        let q_stride = g
            .query_length()
            .checked_mul(g.columns())
            .ok_or("cross Q stride overflow")?;
        let kv_stride = u64::from(packed_rows)
            .checked_mul(g.columns())
            .ok_or("cross KV stride overflow")?;
        Ok(CudnnSdpaSignature {
            batch: i64::from(stage.batches),
            heads: checked(g.heads())?,
            query_length: i64::from(stage.query_count),
            key_value_length: i64::from(packed_rows),
            head_dim: checked(g.head_dim())?,
            query_batch_stride: checked(q_stride)?,
            key_value_batch_stride: checked(kv_stride)?,
            output_batch_stride: checked(q_stride)?,
            scale_bits: g.effective_scale().to_bits(),
        })
    }
}
