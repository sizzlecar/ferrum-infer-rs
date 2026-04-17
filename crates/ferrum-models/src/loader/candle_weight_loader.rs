//! Temporary `WeightLoader<B>` bridge over candle's `VarBuilder`.
//!
//! This is the Phase B/C shim: it lets the new Model-as-Code path
//! (`Qwen3Model<B>` etc.) source weights from the existing
//! candle-safetensors loader that `SafeTensorsLoader::load_varbuilder`
//! already gives us, without waiting for a native-safetensors `WeightLoader`
//! implementation.
//!
//! Native `SafeTensorsLoader : WeightLoader<B>` (no candle dependency) lands
//! in Phase D as part of dropping candle from the hot path.

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use ferrum_kernels::backend::Backend;
use ferrum_quantization::{DenseLinear, Linear, QuantConfig, WeightLoader};
use ferrum_types::{FerrumError, Result};

pub struct CandleVarBuilderLoader<'a, B: Backend> {
    vb: VarBuilder<'a>,
    _m: std::marker::PhantomData<B>,
}

impl<'a, B: Backend> CandleVarBuilderLoader<'a, B> {
    pub fn new(vb: VarBuilder<'a>) -> Self {
        Self {
            vb,
            _m: std::marker::PhantomData,
        }
    }

    fn get_tensor(&self, key: &str) -> Option<Tensor> {
        self.vb.get_unchecked(key).ok()
    }

    fn tensor_to_f32_vec(&self, t: &Tensor) -> Result<Vec<f32>> {
        let t = t
            .to_dtype(DType::F32)
            .map_err(|e| FerrumError::model(format!("to_f32: {e}")))?;
        let t = t
            .flatten_all()
            .map_err(|e| FerrumError::model(format!("flatten: {e}")))?;
        t.to_vec1::<f32>()
            .map_err(|e| FerrumError::model(format!("to_vec: {e}")))
    }

    fn fetch_2d(&self, key: &str) -> Option<(usize, usize, Vec<f32>)> {
        let t = self.get_tensor(key)?;
        let shape = t.shape().dims().to_vec();
        if shape.len() != 2 {
            return None;
        }
        let data = self.tensor_to_f32_vec(&t).ok()?;
        Some((shape[0], shape[1], data))
    }

    fn fetch_2d_concat(&self, keys: &[String]) -> Option<(usize, usize, Vec<f32>)> {
        let mut total_rows = 0usize;
        let mut cols = 0usize;
        let mut acc: Vec<f32> = Vec::new();
        for k in keys {
            let (r, c, data) = self.fetch_2d(k)?;
            if cols == 0 {
                cols = c;
            } else if cols != c {
                return None;
            }
            total_rows += r;
            acc.extend_from_slice(&data);
        }
        Some((total_rows, cols, acc))
    }
}

impl<'a, B: Backend> WeightLoader<B> for CandleVarBuilderLoader<'a, B> {
    fn load_tensor(&self, name: &str) -> Result<B::Buffer> {
        let t = self
            .get_tensor(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not found")))?;
        let data = self.tensor_to_f32_vec(&t)?;
        Ok(B::from_slice(&data))
    }

    fn load_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>> {
        // Try the fused `<name>.weight` tensor first.
        let direct = format!("{name}.weight");
        if let Some((r, c, data)) = self.fetch_2d(&direct) {
            return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
        }

        // Fused-projection aliases: Qwen3 / Llama-family checkpoints split
        // these along dim 0. Concat and wrap in a single DenseLinear.
        if name.ends_with("qkv_proj") {
            let prefix = &name[..name.len() - "qkv_proj".len()];
            let keys = vec![
                format!("{prefix}q_proj.weight"),
                format!("{prefix}k_proj.weight"),
                format!("{prefix}v_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }
        if name.ends_with("gate_up_proj") {
            let prefix = &name[..name.len() - "gate_up_proj".len()];
            let keys = vec![
                format!("{prefix}gate_proj.weight"),
                format!("{prefix}up_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }

        Err(FerrumError::model(format!(
            "could not load linear '{name}' — no direct weight, no split components"
        )))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.vb.get_unchecked(name).is_ok()
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        None
    }
}
