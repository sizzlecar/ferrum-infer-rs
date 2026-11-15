use super::super::program::Graph;
use super::*;

fn snake(
    g: &mut Graph,
    name: &str,
    x: ProgramValueId,
    channels: u64,
    samples: u64,
) -> Result<ProgramValueId, VNextError> {
    g.emit(
        name,
        SNAKE_ALPHA_NCHW_F32_OPERATION_ID,
        vec![x, weight_value(&format!("{name}.alpha"))?],
        channels,
        samples,
        BTreeMap::new(),
    )
}
pub(super) fn build(
    c: &EncoderConfig,
    limits: FastH3AudioComponentLimits,
) -> Result<(ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>), VNextError> {
    let a = &c.architecture;
    let mut samples = c.padded_samples()?;
    let input = id(PreparedFastH3AudioEncoderTrunk::INPUT_ID)?;
    let inputs = BTreeMap::from([(input, tensor(vec![c.mono_batch, 1, 1, samples]))]);
    if inputs.values().next().unwrap().byte_len()? > limits.maximum_activation_bytes {
        return Err(invalid("audio encoder input exceeds capacity"));
    }
    let mut g = Graph::new(c.mono_batch, limits, inputs.clone());
    let mut channels = a.encoder_dim;
    let mut x = g.conv(
        "encoder.block.0",
        id(PreparedFastH3AudioEncoderTrunk::INPUT_ID)?,
        channels,
        samples,
        7,
        1,
    )?;
    for (stage, &stride) in a.encoder_rates.iter().enumerate() {
        let prefix = format!("encoder.block.{}.block", stage + 1);
        for (unit, dilation) in [1, 3, 9].into_iter().enumerate() {
            let residual = format!("{prefix}.{unit}.block");
            let y = snake(
                &mut g,
                &format!("{residual}.0"),
                x.clone(),
                channels,
                samples,
            )?;
            let y = g.conv(&format!("{residual}.1"), y, channels, samples, 7, dilation)?;
            let y = snake(&mut g, &format!("{residual}.2"), y, channels, samples)?;
            let y = g.conv(&format!("{residual}.3"), y, channels, samples, 1, 1)?;
            x = g.add(
                &format!("{prefix}.{unit}.residual"),
                x,
                y,
                channels,
                samples,
            )?;
        }
        x = snake(&mut g, &format!("{prefix}.3"), x, channels, samples)?;
        let pad = stride.div_ceil(2);
        samples = samples
            .checked_add(pad * 2)
            .and_then(|n| n.checked_sub(stride * 2))
            .map(|n| n / stride + 1)
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid("audio encoder convolution extent overflow"))?;
        channels = channels
            .checked_mul(2)
            .ok_or_else(|| invalid("audio encoder channel overflow"))?;
        x = g.convolution(
            &format!("{prefix}.4"),
            &format!("{prefix}.4"),
            x,
            channels,
            samples,
            1,
            stride,
            pad,
        )?;
    }
    x = snake(
        &mut g,
        &format!("encoder.block.{}", a.encoder_rates.len() + 1),
        x,
        channels,
        samples,
    )?;
    x = g.convolution(
        "encoder.trunk",
        &format!("encoder.block.{}", a.encoder_rates.len() + 2),
        x,
        a.latent_dim,
        samples,
        1,
        1,
        1,
    )?;
    g.finish(
        family::Family::<EncoderConfig>::FAMILY_ID,
        "block.fast_h3.audio.encoder.trunk",
        inputs,
        &c.weights,
        x,
    )
}
