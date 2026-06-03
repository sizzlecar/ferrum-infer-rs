//! LoRA reference utilities.
//!
//! G4 keeps production adapter serving startup-scoped. This module provides
//! the small f32 reference path used by loader and routing tests:
//!
//! y = base(x) + (alpha / r) * B(A(x))

use ferrum_types::{FerrumError, Result};

#[derive(Debug, Clone)]
pub struct LoraLinearRef {
    base_weight: Vec<f32>,
    a_weight: Vec<f32>,
    b_weight: Vec<f32>,
    in_features: usize,
    out_features: usize,
    rank: usize,
    scaling: f32,
}

impl LoraLinearRef {
    pub fn new(
        base_weight: Vec<f32>,
        a_weight: Vec<f32>,
        b_weight: Vec<f32>,
        in_features: usize,
        out_features: usize,
        rank: usize,
        lora_alpha: f32,
    ) -> Result<Self> {
        if rank == 0 {
            return Err(FerrumError::config("LoRA rank must be > 0"));
        }
        if base_weight.len() != out_features * in_features {
            return Err(FerrumError::config(format!(
                "base weight shape mismatch: got {} elements, expected {}x{}",
                base_weight.len(),
                out_features,
                in_features
            )));
        }
        if a_weight.len() != rank * in_features {
            return Err(FerrumError::config(format!(
                "LoRA A shape mismatch: got {} elements, expected {}x{}",
                a_weight.len(),
                rank,
                in_features
            )));
        }
        if b_weight.len() != out_features * rank {
            return Err(FerrumError::config(format!(
                "LoRA B shape mismatch: got {} elements, expected {}x{}",
                b_weight.len(),
                out_features,
                rank
            )));
        }
        Ok(Self {
            base_weight,
            a_weight,
            b_weight,
            in_features,
            out_features,
            rank,
            scaling: lora_alpha / rank as f32,
        })
    }

    pub fn forward(&self, input: &[f32], batch: usize) -> Result<Vec<f32>> {
        if input.len() != batch * self.in_features {
            return Err(FerrumError::config(format!(
                "LoRA input shape mismatch: got {} elements, expected {}x{}",
                input.len(),
                batch,
                self.in_features
            )));
        }
        let mut out = vec![0.0f32; batch * self.out_features];
        let mut low_rank = vec![0.0f32; batch * self.rank];

        for m in 0..batch {
            for o in 0..self.out_features {
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += input[m * self.in_features + i]
                        * self.base_weight[o * self.in_features + i];
                }
                out[m * self.out_features + o] = acc;
            }
            for r in 0..self.rank {
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += input[m * self.in_features + i]
                        * self.a_weight[r * self.in_features + i];
                }
                low_rank[m * self.rank + r] = acc;
            }
            for o in 0..self.out_features {
                let mut acc = 0.0f32;
                for r in 0..self.rank {
                    acc += low_rank[m * self.rank + r] * self.b_weight[o * self.rank + r];
                }
                out[m * self.out_features + o] += self.scaling * acc;
            }
        }

        Ok(out)
    }
}
