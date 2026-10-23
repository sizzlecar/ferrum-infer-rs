//! Numerical identities for the independent text-projection primitives.
//!
//! The gamma payload is BF16. Its conversion and the zero-centering addition
//! are FP32 operations, not a change to the stored weight format. GELU calls
//! libdevice tanhf on a computed FP32 argument: qualification of BF16-input
//! tanh does not qualify this profile.

pub const ZERO_CENTERED_RMS_NORM_PROFILE_ID: &str =
    "cuda.ieee.zero_centered_rms_norm.bf16.pairwise_sqrt_div.v1";
pub const GELU_TANH_PROFILE_ID: &str = "cuda.libdevice.precise.gelu_tanh.bf16.v1";

pub const NUMERICAL_IDENTITY: &str = concat!(
    "text-projection-primitives-v1;BF16-input-weight-output;integer-BF16-RNE;",
    "gradual-underflow;explicit-RN32-add-mul-div-sqrt;no-implicit-FMA;",
    "ZeroCenteredRMS=whole-row-power2-width-1..4096;adjacent-pair-square-sum;",
    "mean=RN32(sum/width);inv=RN32(1/sqrtRN32(RN32(mean+epsilon)));",
    "y=B(RN32(RN32(x*inv)*RN32(decode(weight)+1)));no-intermediate-B;",
    "epsilon=(numerator-as-f64/denominator-as-f64)-as-f32;",
    "RMS-dynamic-CTA-shared=4*width;static-CTA-shared=one-FP32-inverse;",
    "GELU-beta-bits=3f4c422a;kappa-bits=3d372713;xx=RN32(x*x);",
    "cube=RN32(xx*x);t=RN32(beta*RN32(x+RN32(kappa*cube)));",
    "y=B(RN32(RN32(0.5*x)*RN32(1+libdevice-tanhf(t))));",
    "GELU-tanh-argument-is-FP32-not-BF16;NaN-class-only;",
    "zero-global-scratch;logical-work-RMS=rows*(5*width+3);GELU=elements*9;",
    "qualification-requires-target-evidence;no-Reference-or-Torch-bitwise-claim"
);
