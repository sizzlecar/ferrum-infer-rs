//! Initial image/audio noise with the Torch CPU MT19937 draw/packing rules.
//!
//! Source: installed Torch 2.14.1, revision
//! `5c4886908584029761b579af026dcfb627c84070`: MT19937RNGEngine.h,
//! TransformationHelper.h (uniform_real), and cpu/DistributionTemplates.h
//! (normal_fill and scalar NormalFill16). Comfy 56f179 calls randn for image,
//! then audio, using the same seeded generator.
//!
//! This is the scalar F32/platform-libm numerical profile, NOT a promise of
//! bitwise equality with Torch's AVX2 log/sincos implementation. Streams shorter
//! than 16 values require a different double/cached-normal path and are rejected.
//! State and output storage are borrowed from the caller's Host accounting.

use anyhow::{ensure, Result};

const STATE_WORDS: usize = 624;
const INDEX_OFFSET: usize = STATE_WORDS * 4;

/// MT state plus its next-word cursor. No temporary normal array is allocated:
/// uniforms and the Box-Muller transform use the caller's output in place.
pub const SCRATCH_BYTES: usize = INDEX_OFFSET + 4;

/// Fill two contiguous F32 little-endian payloads, image first, then audio.
/// Each stream must contain at least 16 values. Validate all storage before
/// changing any bytes; a rejected call publishes neither partial stream.
pub fn fill_initial_noise(
    seed: u64,
    image: &mut [u8],
    audio: &mut [u8],
    scratch: &mut [u8],
) -> Result<()> {
    for (name, bytes) in [("image", image.len()), ("audio", audio.len())] {
        ensure!(
            bytes >= 16 * 4 && bytes % 4 == 0,
            "CPU MT19937 {name} noise requires at least 16 complete F32 values"
        );
    }
    ensure!(
        scratch.len() >= SCRATCH_BYTES,
        "CPU MT19937 scratch requires {SCRATCH_BYTES} bytes"
    );
    let mut generator = Mt19937::seeded(seed, &mut scratch[..SCRATCH_BYTES]);
    generator.fill_normal(image);
    generator.fill_normal(audio);
    Ok(())
}

struct Mt19937<'a> {
    bytes: &'a mut [u8],
}

impl<'a> Mt19937<'a> {
    fn seeded(seed: u64, bytes: &'a mut [u8]) -> Self {
        let mut this = Self { bytes };
        // Torch retains the supplied u64 seed as metadata, but initializes its
        // 32-bit MT recurrence using only the low 32 bits.
        let mut word = seed as u32;
        this.set_word(0, word);
        for index in 1..STATE_WORDS {
            word = 1_812_433_253_u32
                .wrapping_mul(word ^ (word >> 30))
                .wrapping_add(index as u32);
            this.set_word(index, word);
        }
        this.set_word(STATE_WORDS, STATE_WORDS as u32);
        this
    }

    fn word(&self, index: usize) -> u32 {
        let start = index * 4;
        u32::from_le_bytes(self.bytes[start..start + 4].try_into().unwrap())
    }

    fn set_word(&mut self, index: usize, word: u32) {
        let start = index * 4;
        self.bytes[start..start + 4].copy_from_slice(&word.to_le_bytes());
    }

    fn next_u32(&mut self) -> u32 {
        let mut index = self.word(STATE_WORDS) as usize;
        if index == STATE_WORDS {
            // In-place recurrence is intentional: wrapped references consume
            // the already-updated words, including state[0] in the last step.
            for i in 0..STATE_WORDS {
                let pair =
                    (self.word(i) & 0x8000_0000) | (self.word((i + 1) % STATE_WORDS) & 0x7fff_ffff);
                let twist = (pair >> 1) ^ if pair & 1 != 0 { 0x9908_b0df } else { 0 };
                self.set_word(i, self.word((i + 397) % STATE_WORDS) ^ twist);
            }
            index = 0;
        }
        let mut word = self.word(index);
        self.set_word(STATE_WORDS, (index + 1) as u32);
        word ^= word >> 11;
        word ^= (word << 7) & 0x9d2c_5680;
        word ^= (word << 15) & 0xefc6_0000;
        word ^ (word >> 18)
    }

    fn fill_normal(&mut self, output: &mut [u8]) {
        // Torch float uniform uses the LOW 24 bits, not the high bits of MT.
        for slot in output.chunks_exact_mut(4) {
            slot.copy_from_slice(&uniform_f32(self.next_u32()).to_le_bytes());
        }
        for block in output.chunks_exact_mut(16 * 4) {
            normal_16(block);
        }
        if output.len() % (16 * 4) != 0 {
            // Match normal_fill's overlapping tail: consume 16 NEW uniforms
            // and overwrite the last 16 outputs, including their prior values.
            let start = output.len() - 16 * 4;
            let tail = &mut output[start..];
            for slot in tail.chunks_exact_mut(4) {
                slot.copy_from_slice(&uniform_f32(self.next_u32()).to_le_bytes());
            }
            normal_16(tail);
        }
    }
}

fn uniform_f32(word: u32) -> f32 {
    (word & 0x00ff_ffff) as f32 * (1.0_f32 / 16_777_216.0)
}

fn normal_16(block: &mut [u8]) {
    for pair in 0..8 {
        let first = pair * 4;
        let second = (pair + 8) * 4;
        let u1 = 1.0_f32 - read_f32(&block[first..first + 4]);
        let u2 = read_f32(&block[second..second + 4]);
        let radius = (-2.0_f32 * u1.ln()).sqrt();
        // The scalar Torch expression uses pi<double>, then stores theta in
        // float. AVX2 instead multiplies by an already-rounded float 2*pi.
        let theta = (std::f64::consts::TAU * f64::from(u2)) as f32;
        let cosine = (radius * theta.cos()).mul_add(1.0, 0.0);
        let sine = (radius * theta.sin()).mul_add(1.0, 0.0);
        block[first..first + 4].copy_from_slice(&cosine.to_le_bytes());
        block[second..second + 4].copy_from_slice(&sine.to_le_bytes());
    }
}

fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().unwrap())
}

#[cfg(test)]
mod tests;
