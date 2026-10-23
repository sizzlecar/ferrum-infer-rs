//! The Reference BF16 transcendental numerical profile and providers.
//!
//! Identity specifies the implementation and arithmetic path, not a claim that
//! numerical qualification has passed. Full-domain qualification lives in the
//! independent test-only oracle. Composition explicitly registers each provider.
//! Cargo pins the dependencies. Updating their versions, sources or arithmetic
//! requires a new profile identity and renewed qualification.

mod provider;
pub(super) use provider::{ReferenceTranscendentalKind, ReferenceTranscendentalProvider};

pub(super) const NUMERICAL_PROFILE_ID: &str = "reference.bf16.transcendentals.libm-0.2.15.v1";

/// Stable dependency/source and arithmetic identity for provider fingerprints.
///
/// These hashes identify the audited registry sources, not their correctness.
/// Normal Cargo checksum verification and exact dependency constraints govern
/// source resolution; product code does not inspect a local Cargo.lock or cache.
pub(super) const NUMERICAL_PROFILE_IDENTITY: &str = concat!(
    "profile=reference.bf16.transcendentals.libm-0.2.15.v1\n",
    "libm.version=0.2.15\n",
    "libm.package.sha256=f9fbbcab51052fe104eb5e5d351cf728d30a5be1fe14d9be8a3b097481fb97de\n",
    "libm.expf.sha256=bbae85acadd2f2a0b7facbf591551f52c936420bb6779672c609cbfd8de0900f\n",
    "libm.tanhf.sha256=bd7879cb81870de06e1b4ceaf0e0e5f5a82f63b618b687cd899306b976bd68a2\n",
    "libm.expm1f.sha256=3c71481940f9d0b51dd035e497378261f150076f33bef85de860bea35dfcde32\n",
    "libm.generic.scalbn.sha256=84a6a3b08af119c8f5f803feb82e5df64622106f550aa3795f61db0b5329dd2c\n",
    "half.version=2.7.1\n",
    "half.package.sha256=6ea2d84b969582b4b1864a92dc5d27cd2b77b622a8d79306834f1be5ba20d84b\n",
    "half.bfloat.sha256=edfc793379e07cf0ffab8e0605608a50facfd41c04b8501a3c878a9e7f05e2c8\n",
    "half.bfloat.convert.sha256=d05c02225e8f44a2d41132c1babf9ec13ea111e0125ca95a0ea31f8889a18384\n",
    "arithmetic=IEEE-FP32-RNE;gradual-underflow;no-FTZ-DAZ;no-cross-boundary-fusion\n",
    "bf16=RNE;NaN-class-only;signed-zero-and-infinity\n",
    "silu=e:libm.expf(-x);d:RN32(1+e);a:BF16(RN32(x/d))\n",
    "silu_mul=a:silu(gate);BF16(RN32(decode(a)*decode(up)))\n",
    "tanh=BF16(libm.tanhf(x))\n",
    "qualification=finite-class-preserving-four-FP32-neighbor-steps;exp-abs-le-2^-25-exact-one\n",
    "logical-work-per-element=silu:4;silu_mul:5;tanh:1;conversions-excluded\n",
);

/// Dependency identity and the actual scalar implementation both contribute.
/// Providers will further distinguish their operation/estimator identities.
pub(super) fn numerical_implementation_fingerprint() -> String {
    super::dense_linear::implementation_fingerprint(&[
        NUMERICAL_PROFILE_ID.as_bytes(),
        NUMERICAL_PROFILE_IDENTITY.as_bytes(),
        include_str!("transcendental.rs").as_bytes(),
        include_str!("runtime/transcendental.rs").as_bytes(),
    ])
}

#[cfg(test)]
mod oracle;
