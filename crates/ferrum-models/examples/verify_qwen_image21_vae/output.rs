//! VAE `_decode` clamps [-1,1]. Diffusers v0.40.0 VaeImageProcessor then
//! denormalizes in the incoming tensor dtype, converts to F32 NHWC, multiplies
//! by 255 and rounds ties-to-even. All four channels, including alpha, follow
//! the same mapping; no RGB conversion, premultiplication or background blend.
//! https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/image_processor.py
use super::artifacts::{self, Artifacts};
use super::png_output;
use anyhow::{ensure, Context, Result};
use ferrum_execution::TensorExecutionOutput;
use half::bf16;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs::OpenOptions, io::Write};

pub fn rgba8(value: bf16) -> Result<u8> {
    ensure!(
        value.is_finite(),
        "nonfinite decoder output cannot become a pixel"
    );
    let clamped = value.to_f32().clamp(-1.0, 1.0);
    let scaled = bf16::from_f32(clamped * 0.5);
    let shifted = bf16::from_f32(scaled.to_f32() + 0.5)
        .to_f32()
        .clamp(0.0, 1.0);
    Ok((shifted * 255.0).round_ties_even() as u8)
}
pub fn save(output: &TensorExecutionOutput, artifacts: &Artifacts) -> Result<Value> {
    let shape = output.tensor().dimensions();
    ensure!(
        shape.len() == 4 && shape[1] == 4,
        "decoder output must be NCHW RGBA"
    );
    let raw = output.bytes().bytes();
    let plane = usize::try_from(
        shape[2]
            .checked_mul(shape[3])
            .context("RGBA plane overflow")?,
    )?;
    let mut nonfinite = 0_u64;
    let mut minimum = f32::INFINITY;
    let mut maximum = f32::NEG_INFINITY;
    for bytes in raw.chunks_exact(2) {
        let x = bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
        if x.is_finite() {
            minimum = minimum.min(x);
            maximum = maximum.max(x);
        } else {
            nonfinite += 1;
        }
    }
    // Preserve the diagnostic raw result even when pixel conversion is invalid.
    artifacts.bytes("rgba.bf16", raw)?;
    let raw_result = json!({"file":"rgba.bf16","shape":shape,"dtype":"BF16","layout":"NCHW","byte_order":"little_endian","sha256":artifacts::sha256(raw),"nonfinite":nonfinite,"minimum":minimum.is_finite().then_some(minimum),"maximum":maximum.is_finite().then_some(maximum)});
    if nonfinite != 0 {
        return Ok(
            json!({"passed":false,"raw":raw_result,"rgba8":[],"qualification":"execution/finite smoke only"}),
        );
    }
    let mut pixels = Vec::new();
    let mut pngs = Vec::new();
    for b in 0..usize::try_from(shape[0])? {
        let name = format!("rgba-{b}.rgba8");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(artifacts.directory().join(&name))?;
        let mut scratch = [0_u8; 16384];
        let mut filled = 0;
        let mut hash = Sha256::new();
        let png = png_output::write(
            &artifacts.directory().join(format!("rgba-{b}.png")),
            shape[3],
            shape[2],
            |stream| {
                for p in 0..plane {
                    for c in 0..4 {
                        let index = ((b * 4 + c) * plane + p) * 2;
                        scratch[filled] = rgba8(bf16::from_bits(u16::from_le_bytes([
                            raw[index],
                            raw[index + 1],
                        ])))?;
                        filled += 1;
                    }
                    if filled == scratch.len() {
                        file.write_all(&scratch)?;
                        stream.write_all(&scratch)?;
                        hash.update(scratch);
                        filled = 0;
                    }
                }
                file.write_all(&scratch[..filled])?;
                stream.write_all(&scratch[..filled])?;
                hash.update(&scratch[..filled]);
                Ok(())
            },
        )?;
        file.sync_all()?;
        let pixel_hash = format!("{:x}", hash.finalize());
        ensure!(
            png.decoded_rgba8_sha256 == pixel_hash,
            "decoded PNG pixels differ from the RGBA8 artifact"
        );
        pixels.push(json!({"file":name,"shape":[shape[2],shape[3],4],"layout":"HWC_RGBA","dtype":"u8","sha256":pixel_hash}));
        pngs.push(png);
    }
    Ok(
        json!({"passed":true,"raw":raw_result,"rgba8":pixels,"postprocess":"clamp[-1,1]; B(B(x*0.5)+0.5); clamp[0,1]; F32*255; ties-even u8; alpha retained, no compositing","png":pngs,"qualification":"execution/finite smoke only; not an independent full-decoder numerical oracle or image-quality qualification"}),
    )
}
