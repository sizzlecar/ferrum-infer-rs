use super::{arguments::Arguments, artifacts};
use anyhow::{bail, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21ModelTimestep, QwenImage21Timestep256, QwenImage21TransformerConfig,
    QwenImage21TransformerFamily as Family,
};
use ferrum_runtime::{ProgramValueId, ResolvedTensorSpec};

pub struct Inputs {
    pub bytes: [Vec<u8>; 4],
    pub model_timesteps: Vec<QwenImage21ModelTimestep>,
}
impl Inputs {
    pub fn prepare(config: &QwenImage21TransformerConfig, args: &Arguments) -> Result<Self> {
        let specs = config.input_tensor_specs()?;
        let lengths = Family::INPUT_IDS
            .map(|name| {
                let spec = &specs[&ProgramValueId::new(name).expect("constant ID")];
                ResolvedTensorSpec::new(
                    spec.dimensions.clone(),
                    spec.element_type,
                    spec.layout.clone(),
                )
                .and_then(|tensor| tensor.minimum_storage_bytes())
            })
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let total = lengths
            .iter()
            .try_fold(0_u64, |sum, length| sum.checked_add(*length))
            .context("input byte sum overflows")?;
        if total > args.maximum_input_bytes as u64 {
            bail!("all four input payloads need {total} bytes, exceeding --maximum-input-bytes before reading or allocation");
        }
        let latent = read_bf16(&args.latent_file, usize::try_from(lengths[0])?)?;
        let text = read_bf16(&args.text_file, usize::try_from(lengths[1])?)?;
        let model_timesteps = args
            .scheduler_timesteps
            .iter()
            .map(|value| QwenImage21ModelTimestep::from_scheduler_timestep(*value))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut time = vec![0_u16; usize::try_from(lengths[2] / 2)?];
        QwenImage21Timestep256::new()?.write_into(config, &model_timesteps, &mut time)?;
        let mut frequencies = vec![0_f32; usize::try_from(lengths[3] / 4)?];
        config
            .rotary_frequencies()?
            .write_into(config.layout(), &mut frequencies)?;
        let time = time.into_iter().flat_map(u16::to_le_bytes).collect();
        let frequencies = frequencies.into_iter().flat_map(f32::to_le_bytes).collect();
        Ok(Self {
            bytes: [latent, text, time, frequencies],
            model_timesteps,
        })
    }
}

fn read_bf16(path: &std::path::Path, length: usize) -> Result<Vec<u8>> {
    let bytes = artifacts::read_bounded(path, length)?;
    if bytes.len() != length {
        bail!(
            "{} does not match its exact typed BF16 shape",
            path.display()
        );
    }
    if bytes
        .chunks_exact(2)
        .any(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) & 0x7f80 == 0x7f80)
    {
        bail!("{} contains nonfinite BF16 input", path.display());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn supplied_bf16_payload_must_be_exact_and_finite() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("input.bf16");
        std::fs::write(&file, [0_u8, 0x3f, 0, 0xbf]).unwrap();
        assert_eq!(read_bf16(&file, 4).unwrap().len(), 4);
        assert!(read_bf16(&file, 2).is_err());
        assert!(read_bf16(&file, 6).is_err());
        for bad in [0x7f80_u16, 0xff80, 0x7fc1] {
            std::fs::write(&file, bad.to_le_bytes()).unwrap();
            assert!(read_bf16(&file, 2).is_err());
        }
    }
}
