//! Independent scalar test oracle: published Random123 counter arithmetic and
//! a host libm Box-Muller calculation with explicit F32 boundaries. Native
//! cuRAND uses device log/sincos approximations: comparisons use a declared
//! absolute/relative bound, never a Torch CPU or bitwise-libm claim.
use ferrum_runtime::PhiloxSeedOffset;

fn counter_words(mut counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    for round in 0..10_u32 {
        let a = u64::from(counter[0]) * 0xD251_1F53;
        let b = u64::from(counter[2]) * 0xCD9E_8D57;
        counter = [
            (b >> 32) as u32 ^ counter[1] ^ key[0].wrapping_add(round.wrapping_mul(0x9E37_79B9)),
            b as u32,
            (a >> 32) as u32 ^ counter[3] ^ key[1].wrapping_add(round.wrapping_mul(0xBB67_AE85)),
            a as u32,
        ];
    }
    counter
}

pub(super) fn known_answers() {
    // Published independent integer KATs:
    // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    assert_eq!(
        counter_words([0; 4], [0; 2]),
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    );
    assert_eq!(
        counter_words([u32::MAX; 4], [u32::MAX; 2]),
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
    );
    assert_eq!(
        counter_words(
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0]
        ),
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
    );
}

/// cuRAND state uses a low 64-bit block count and a high 64-bit subsequence.
/// Integer ordinal arithmetic is deliberately u128, independent of the
/// native state increment/carry implementation. Unaligned offsets draw four
/// consecutive words across blocks before the two Box-Muller pairs.
pub(super) fn normal4(state: PhiloxSeedOffset, subsequence: u64, round: u64) -> [f32; 4] {
    let key = [state.seed() as u32, (state.seed() >> 32) as u32];
    let mut words = [0; 4];
    for (lane, word) in words.iter_mut().enumerate() {
        let ordinal = u128::from(state.offset()) + u128::from(round) * 4 + lane as u128;
        let counter = (u128::from(subsequence) << 64).wrapping_add(ordinal / 4);
        *word = counter_words(
            [
                counter as u32,
                (counter >> 32) as u32,
                (counter >> 64) as u32,
                (counter >> 96) as u32,
            ],
            key,
        )[(ordinal % 4) as usize];
    }
    // NVIDIA documents native cuRAND's pair ordering and uniform centering:
    // https://docs.nvidia.com/cuda/curand/curand/curand__normal_8h_source.html
    // This implements the mathematical transform; no NVIDIA source is copied.
    let mut output = [0.; 4];
    for pair in 0..2 {
        let uniform = words[2 * pair] as f32 * 2_f32.powi(-32) + 2_f32.powi(-33);
        let angle_step = std::f32::consts::TAU * 2_f32.powi(-32);
        let angle = words[2 * pair + 1] as f32 * angle_step + angle_step / 2.;
        let logarithm = libm::log(f64::from(uniform)) as f32;
        let radius = libm::sqrt(f64::from(-2. * logarithm)) as f32;
        output[2 * pair] = (libm::sin(f64::from(angle)) as f32 * radius) + 0.;
        output[2 * pair + 1] = (libm::cos(f64::from(angle)) as f32 * radius) + 0.;
    }
    output
}

pub(super) fn check(state: PhiloxSeedOffset, blocks: u32, actual: &[f32]) {
    let stride = u64::from(blocks) * 256;
    assert!(stride > 0);
    for (index, &got) in actual.iter().enumerate() {
        let index = index as u64;
        let round = index / (4 * stride);
        let component = (index / stride) % 4;
        let thread = index % stride;
        let expected = normal4(state, thread, round)[component as usize];
        // Before device execution: bounds the different libm/device log and
        // approximate native sincos implementations. Much narrower than BF16
        // precision; direct F32 preservation is also tested by exact BF16 RNE.
        let bound = 2e-5_f32 * (1. + expected.abs());
        assert!(got.is_finite() && (got-expected).abs() <= bound,
            "seed={} offset={} index={index} thread={thread} round={round} component={component}: got={got} oracle={expected} bound={bound}",state.seed(),state.offset());
    }
}

#[test]
fn published_counter_vectors_and_untruncated_f32_normal_oracle() {
    known_answers();
    let state = PhiloxSeedOffset::fresh(0);
    let values = normal4(state, 0, 0);
    assert!(values.iter().all(|x| x.is_finite()));
    assert!(
        values.iter().all(|x| x.to_bits() & 0xffff != 0),
        "oracle must retain bits below BF16"
    );
    let promoted: Vec<f32> = values
        .iter()
        .map(|&x| half::bf16::from_f32(x).to_f32())
        .collect();
    assert!(
        values
            .iter()
            .zip(promoted)
            .any(|(&x, b)| (x - b).abs() > 2e-5 * (1. + x.abs())),
        "oracle must detect BF16-promote substitution"
    );
    for offset in [1, 3, 4, (1_u64 << 40) + 3, u64::MAX] {
        assert_ne!(normal4(PhiloxSeedOffset::new(0, offset), 0, 0), values);
    }
    assert_ne!(normal4(PhiloxSeedOffset::fresh(1_u64 << 40), 0, 0), values);
    assert_ne!(normal4(state, 1_u64 << 40, 0), values);
    let expected = normal4(PhiloxSeedOffset::new(7, 3), 4, 1);
    assert_eq!(expected, normal4(PhiloxSeedOffset::new(7, 7), 4, 0));
    let mut arranged = vec![0.; 1025];
    for (i, x) in arranged.iter_mut().enumerate() {
        let stride = 256;
        *x = normal4(state, (i % stride) as u64, (i / (stride * 4)) as u64)[(i / stride) % 4];
    }
    check(state, 1, &arranged);
}
