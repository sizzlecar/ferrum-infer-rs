//! GELU(tanh) scalar with a single BF16 boundary and explicit FP32 stages.
use half::bf16;
/// Constants are fixed FP32 values, not platform runtime sqrt/pi evaluations.
const BETA: f32 = f32::from_bits(0x3f4c422a);
const KAPPA: f32 = f32::from_bits(0x3d372713);
/// Nine logical operations: five polynomial arithmetic operations, tanh,
/// multiplication by 0.5, addition of 1, and final multiplication. Conversion
/// is excluded. This is not the number of hardware instructions/FLOPs.
pub(in crate::backend::reference) fn gelu_tanh_bf16_bits(raw: u16) -> u16 {
    let x = bf16::from_bits(raw).to_f32();
    let square = x * x;
    let cube = square * x;
    let cubic_term = KAPPA * cube;
    let polynomial = x + cubic_term;
    let inner = BETA * polynomial;
    let activation = libm::tanhf(inner);
    let left = 0.5_f32 * x;
    let right = 1.0_f32 + activation;
    bf16::from_f32(left * right).to_bits()
}
#[cfg(test)]
mod tests;
