use super::*;

#[test]
fn mt19937_matches_published_words_and_wraps_many_state_cycles() {
    let mut scratch = [0_u8; SCRATCH_BYTES];
    let mut generator = Mt19937::seeded(5489, &mut scratch);
    // MT19937's standard default-seed reference sequence, independent of the
    // normal transform, checks initialization, tempering and state renewal.
    let expected = [
        3_499_211_612,
        581_869_302,
        3_890_346_734,
        3_586_334_585,
        545_404_204,
        4_161_255_391,
        3_922_919_429,
        949_333_985,
        2_715_962_298,
        1_323_567_403,
    ];
    for word in expected {
        assert_eq!(generator.next_u32(), word);
    }
    for _ in 10..9999 {
        generator.next_u32();
    }
    assert_eq!(generator.next_u32(), 4_123_659_995);
}

#[test]
fn uniform_mapping_and_seed_width_match_torch_cpu_contract() {
    assert_eq!(uniform_f32(0x0100_0000).to_bits(), 0.0_f32.to_bits());
    assert_eq!(uniform_f32(0x0000_0001), 2.0_f32.powi(-24));
    assert_eq!(uniform_f32(u32::MAX), 1.0_f32 - 2.0_f32.powi(-24));
    let mut a = [0_u8; 64];
    let mut b = [0_u8; 64];
    let mut c = [0_u8; 64];
    let mut d = [0_u8; 64];
    let mut scratch = [0_u8; SCRATCH_BYTES];
    fill_initial_noise(42, &mut a, &mut b, &mut scratch).unwrap();
    fill_initial_noise((1_u64 << 32) + 42, &mut c, &mut d, &mut scratch).unwrap();
    assert_eq!(a, c);
    assert_eq!(b, d);
    assert_ne!(a, b, "audio must continue the generator, not reseed");
}

fn oracle_pair(first: f32, second: f32) -> (f64, f64) {
    let radius = (-2.0 * (1.0 - f64::from(first)).ln()).sqrt();
    let theta = std::f64::consts::TAU * f64::from(second);
    (radius * theta.cos(), radius * theta.sin())
}

fn close(actual: f32, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (f64::from(actual) - expected).abs() <= 1e-6 * expected.abs().max(1.0),
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn overlapping_tail_consumes_fresh_uniforms_before_audio_and_preserves_guards() {
    // Unaligned byte slices exercise the actual LE ABI; no typed pointer cast
    // or extra alignment requirement may escape the caller's Host allocation.
    let mut image = [0xa5_u8; 17 * 4 + 2];
    let mut audio = [0x5a_u8; 16 * 4 + 2];
    let mut state = [0xcc_u8; SCRATCH_BYTES + 2];
    fill_initial_noise(
        42,
        &mut image[1..1 + 17 * 4],
        &mut audio[1..1 + 16 * 4],
        &mut state[1..1 + SCRATCH_BYTES],
    )
    .unwrap();
    assert_eq!((image[0], image[image.len() - 1]), (0xa5, 0xa5));
    assert_eq!((audio[0], audio[audio.len() - 1]), (0x5a, 0x5a));
    assert_eq!((state[0], state[state.len() - 1]), (0xcc, 0xcc));

    let mut reference_state = [0_u8; SCRATCH_BYTES];
    let mut generator = Mt19937::seeded(42, &mut reference_state);
    let uniforms: Vec<_> = (0..49).map(|_| uniform_f32(generator.next_u32())).collect();
    // Only image[0] survives its first 16-value transform. The tail replaces
    // image[1..17] with draws[17..33]; audio starts at draw 33, not draw 17.
    close(
        read_f32(&image[1..5]),
        oracle_pair(uniforms[0], uniforms[8]).0,
    );
    for pair in 0..8 {
        let expected = oracle_pair(uniforms[17 + pair], uniforms[25 + pair]);
        close(read_f32(&image[5 + pair * 4..9 + pair * 4]), expected.0);
        close(read_f32(&image[37 + pair * 4..41 + pair * 4]), expected.1);
        let expected = oracle_pair(uniforms[33 + pair], uniforms[41 + pair]);
        close(read_f32(&audio[1 + pair * 4..5 + pair * 4]), expected.0);
        close(read_f32(&audio[33 + pair * 4..37 + pair * 4]), expected.1);
    }
    assert_eq!(&state[1..1 + SCRATCH_BYTES], &reference_state);
}

#[test]
fn whole_block_stream_boundaries_preserve_one_request_local_sequence() {
    let mut first = [0_u8; 48 * 4];
    let mut second = [0_u8; 48 * 4];
    let mut scratch = [0_u8; SCRATCH_BYTES];
    let (image, audio) = first.split_at_mut(16 * 4);
    fill_initial_noise(1234, image, audio, &mut scratch).unwrap();
    let (image, audio) = second.split_at_mut(32 * 4);
    fill_initial_noise(1234, image, audio, &mut scratch).unwrap();
    assert_eq!(first, second);
}

#[test]
fn rejected_extents_leave_outputs_and_state_untouched() {
    for (image_bytes, audio_bytes, scratch_bytes) in [
        (63, 64, SCRATCH_BYTES),
        (64, 60, SCRATCH_BYTES),
        (65, 64, SCRATCH_BYTES),
        (64, 64, SCRATCH_BYTES - 1),
    ] {
        let mut image = vec![0xa5; image_bytes];
        let mut audio = vec![0x5a; audio_bytes];
        let mut scratch = vec![0xcc; scratch_bytes];
        assert!(fill_initial_noise(42, &mut image, &mut audio, &mut scratch).is_err());
        assert!(image.iter().all(|&byte| byte == 0xa5));
        assert!(audio.iter().all(|&byte| byte == 0x5a));
        assert!(scratch.iter().all(|&byte| byte == 0xcc));
    }
}
