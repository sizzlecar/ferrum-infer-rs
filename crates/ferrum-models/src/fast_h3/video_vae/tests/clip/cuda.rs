use super::super::cuda_support::{self, id};
use super::*;

/// Deliberately zero block residual scales isolate the real outer arithmetic.
/// The separate complete-block fixture covers nonzero attention/FFN/residuals.
/// Here every original block still compiles, uploads and executes on CUDA.
fn outer_fixture() -> Fixture {
    let mut f = Fixture::new();
    for (name, (_, _, data)) in &mut f.tensors {
        if name.ends_with(".weight_global_sf") {
            *data = 1f32.to_le_bytes().to_vec();
        }
    }
    f.tensors.get_mut("post_quant_conv.weight").unwrap().2 = [1f32, 0., 0., 1.]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let weights = &mut f.tensors.get_mut("decoder.proj_in.weight").unwrap().2;
    for row in 0..128 {
        weights[4 * (2 * row + row % 2)..4 * (2 * row + row % 2) + 4]
            .copy_from_slice(&1f32.to_le_bytes());
    }
    f.tensors.get_mut("decoder.norm_out.weight").unwrap().2 = 1f32.to_le_bytes().repeat(128);
    let weights = &mut f.tensors.get_mut("decoder.proj_out.weight").unwrap().2;
    for row in 0..48 {
        weights[4 * (128 * row + row)..4 * (128 * row + row) + 4]
            .copy_from_slice(&1f32.to_le_bytes());
    }
    f
}

#[test]
#[ignore = "requires locked SM120 NVFP4 plus CUDA tensor module; full graph exact outer subset, not general Torch parity"]
fn full_video_clip_runs_original_outer_blocks_and_managed_ncthw_unpack() {
    let (_dir, package) = outer_fixture().open(limits());
    let account = cuda_support::host();
    let (config, source) = FastH3VideoClipSource::prepare(
        Arc::new(package.unwrap()),
        [2, 2, 1, 2],
        &account,
        clip_limits(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(clip_limits()).unwrap())
        .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(F::PROFILE_ID))
        .unwrap();
    // B0 channels (+1,-1), B1 (-1,+1), two time steps and two spatial tiles.
    let dense = [vec![1f32; 4], vec![-1.; 4], vec![-1.; 4], vec![1.; 4]].concat();
    let latent = config
        .managed_input_from_decoder_ncthw(&dense, &account, clip_limits())
        .unwrap();
    let (cosine, sine) = config
        .managed_bf16_frequencies(&account, clip_limits())
        .unwrap();
    let retained = cuda_support::execute_twice(
        prepared,
        config.compile_options(clip_limits()).unwrap(),
        Arc::new(source),
        vec![latent, cosine, sine],
        &account,
        |outputs| {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].value_id().as_str(), F::OUTPUT_ID);
            for (row, patch) in outputs[0].bytes().bytes().chunks_exact(48 * 2).enumerate() {
                for (column, word) in patch.chunks_exact(2).enumerate() {
                    let positive = (column % 2 == 0) == (row < 4);
                    assert_eq!(
                        u16::from_le_bytes([word[0], word[1]]),
                        if positive { 0x3f80 } else { 0xbf80 },
                        "patch {row}, column {column}"
                    );
                }
            }
        },
    );
    // Read after session close. Inverse patch layout is checked independently
    // as [C,pt,ph,pw], including batch/time separation and unequal spatial axes.
    let raw = config
        .unpack_execution(&retained[0], &account, clip_limits())
        .unwrap();
    assert_eq!(raw.ncthw_shape(), [2, 3, 8, 2, 4]);
    for (index, word) in raw.bytes().bytes().chunks_exact(2).enumerate() {
        let batch = index / (3 * 8 * 2 * 4);
        let x = index % 4;
        let positive = (x % 2 == 0) == (batch == 0);
        assert_eq!(
            u16::from_le_bytes([word[0], word[1]]),
            if positive { 0x3f80 } else { 0xbf80 }
        );
    }
    drop(retained);
    assert_eq!(
        account.usage().unwrap().committed_bytes(),
        raw.bytes().len()
    );
    drop(raw);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    eprintln!("Full declared-depth video clip: two batches × temporal2 × height1 × width2, full block executed, original source components, two executions, exact independent outer oracle, BF16 NCTHW unpack and retained-output retirement");
}
