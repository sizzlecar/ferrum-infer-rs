use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21AttentionConfig, QwenImage21AttentionLayout, QwenImage21Block0Broadcast,
    QwenImage21Block0Config, QwenImage21Block0Family as Family, QwenImage21Block0MlpFamily as Mlp,
    QwenImage21Config,
};
use serde_json::{json, Value};

pub fn config(transformer: QwenImage21Config) -> Result<QwenImage21Block0Config> {
    Ok(QwenImage21Block0Config::new(
        QwenImage21AttentionConfig::new(
            transformer,
            QwenImage21AttentionLayout::uniform(2, true)?,
        )?,
        QwenImage21Block0Broadcast::Elementwise,
    )?)
}

pub fn declaration() -> Value {
    json!({"batch":1,"sequence":2,"visibility":"both rows attend both keys","broadcast":"elementwise",
        "x":"row0 alternates +1,-1 by channel; row1 is its negative",
        "scale1":0.25,"scale2":-0.125,"gate1":0.5,"gate2":-0.25,
        "gate_interpretation":"raw values: the graph applies tanh before each residual",
        "frequencies":"all adjacent pairs have F32 cos=1,sin=+0",
        "qualification":"fixed nonzero diagnostic probe, not an upstream-conditioning sample or a benchmark"})
}

pub struct Inputs {
    pub bytes: [Vec<u8>; 6],
}
impl Inputs {
    pub fn new(config: &QwenImage21Block0Config) -> Result<Self> {
        if config.attention().layout().batch() != 1
            || config.attention().layout().sequence_length() != 2
            || config.broadcast() != QwenImage21Block0Broadcast::Elementwise
        {
            bail!("fixed block probe requires B1/S2 Elementwise geometry");
        }
        let specs = config.input_tensor_specs()?;
        let mut bytes: [Vec<u8>; 6] = std::array::from_fn(|_| Vec::new());
        for (index, name) in Family::INPUT_IDS.iter().enumerate() {
            let spec = &specs[&ferrum_runtime::ProgramValueId::new(*name)?];
            let count = usize::try_from(spec.byte_len()?)?;
            bytes[index]
                .try_reserve_exact(count)
                .context("probe input capacity")?;
            if index == 5 {
                for _ in 0..count / 8 {
                    bytes[index].extend_from_slice(&1_f32.to_le_bytes());
                    bytes[index].extend_from_slice(&0_f32.to_le_bytes());
                }
            } else {
                let width = usize::try_from(config.transformer().hidden_size())?;
                for offset in 0..count / 2 {
                    let bits: u16 = match index {
                        0 => {
                            if (offset % width + offset / width) % 2 == 0 {
                                0x3f80
                            } else {
                                0xbf80
                            }
                        }
                        1 => 0x3e80, // scale1 = 1/4
                        2 => 0x3f00, // gate1 = 1/2
                        3 => 0xbe00, // scale2 = -1/8
                        4 => 0xbe80, // gate2 = -1/4
                        _ => unreachable!(),
                    };
                    bytes[index].extend_from_slice(&bits.to_le_bytes());
                }
            }
            if bytes[index].len() != count {
                bail!("fixed probe did not fill its exact input shape");
            }
        }
        Ok(Self { bytes })
    }
}

#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub value_id: String,
    pub file: String,
    pub shape: Vec<u64>,
    pub byte_len: usize,
}
pub fn output_specs(config: &QwenImage21Block0Config) -> Result<Vec<OutputSpec>> {
    Family::OUTPUT_IDS
        .iter()
        .enumerate()
        .map(|(index, name)| {
            let width = if [Mlp::GATE_ID, Mlp::UP_ID, Mlp::ACTIVATION_ID].contains(name) {
                config.transformer().mlp_hidden_size()
            } else {
                config.transformer().hidden_size()
            };
            let rows = config.attention().layout().rows();
            let byte_len = rows
                .checked_mul(width)
                .and_then(|n| n.checked_mul(2))
                .context("output shape overflows")?;
            let byte_len = usize::try_from(byte_len)?;
            if byte_len > isize::MAX as usize {
                bail!("output byte length exceeds addressable storage");
            }
            Ok(OutputSpec {
                value_id: (*name).into(),
                file: format!("output-{index:02}.bf16"),
                shape: vec![rows, width],
                byte_len,
            })
        })
        .collect()
}
