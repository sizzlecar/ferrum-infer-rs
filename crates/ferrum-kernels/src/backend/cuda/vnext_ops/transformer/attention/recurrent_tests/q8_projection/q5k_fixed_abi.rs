//! Strict Q5K control/candidate composition through real GDN GPU state updates.
use super::*;

#[test]
#[ignore = "requires an actual CUDA device; composed strict GDN output/state equivalence"]
fn recurrent_q5k_smallrow_matches_generic_projection_output_and_carried_state_bits() {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.gdn-q5k-fixed-abi").unwrap(),
            AttentionExecutionPolicy::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let provider = CudaGatedDeltaRecurrentAttentionProvider::new_f32_master(&runtime).unwrap();
    let mut control = provider.functions.clone();
    control.native = control.native.with_generic_q5k_control();
    let stream = runtime.context().default_stream();
    let q8 = Q8F32ScaleKernels::load(runtime.context()).unwrap();
    // Same mixed formats and existing finite fixture scales as the composed
    // strict policy oracle. Q8 workspace is retained but this test never packs
    // or quantizes activations; both arms call the strict production path.
    let matrices = [
        Matrix::new(&stream, GgufBlockFormat::Q5K, 512, 0, 0, 0),
        Matrix::new(&stream, GgufBlockFormat::Q4K, 256, 512, 1, 0),
        Matrix::new(&stream, GgufBlockFormat::Q8_0, 2, 768, 2, 0),
        Matrix::new(&stream, GgufBlockFormat::Q8_0, 2, 770, 3, 0),
    ];
    let out = [Matrix::new(&stream, GgufBlockFormat::Q5K, HIDDEN, 0, 4, 6)];
    // B1, B4, row-tile tail, last specialized row, then the unchanged boundary.
    // Logical owners have unequal histories and an inactive owner/physical slot.
    let batches = [[0, 0, 1], [0, 1, 3], [0, 3, 4], [0, 7, 24], [0, 8, 24]];
    let old = frames_with_batches(&stream, &control, &q8, &matrices, shape(), false, &batches);
    let new = frames_with_batches(
        &stream,
        &provider.functions,
        &q8,
        &matrices,
        shape(),
        false,
        &batches,
    );
    for (old, new) in old.iter().zip(&new) {
        assert_eq!(
            old.qkv
                .read(&stream)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            new.qkv
                .read(&stream)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "actual GPU QKV including unchanged Q4K/Q8 leaves"
        );
    }
    let old = compose_with_batches(&stream, &control, &q8, &old, &out, false, &batches);
    let new = compose_with_batches(
        &stream,
        &provider.functions,
        &q8,
        &new,
        &out,
        false,
        &batches,
    );
    let bits = |x: &[Vec<f32>]| x.iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&old.0), bits(&new.0), "all GPU recurrent core outputs");
    assert_eq!(
        bits(&old.1),
        bits(&new.1),
        "carried F32 state including idle physical slot"
    );
    assert_eq!(
        old.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        new.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "gated norm, output Q5K projection and F32 residual"
    );
}
