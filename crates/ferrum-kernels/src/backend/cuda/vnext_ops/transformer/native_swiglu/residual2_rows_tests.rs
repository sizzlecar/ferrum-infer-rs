//! Production FFN launcher coverage, not a model-quality or admission receipt.
//! The whole selector is evaluated once; leaf partitions never reselect math.
use super::super::super::test_support::Guarded;
use super::super::route_selection::{select, Arithmetic};
use super::*;
use cudarc::driver::sys;

#[test]
#[ignore = "requires exclusive CUDA; complete production residual2 FFN, guards and graph replay"]
fn residual2_m2to8_complete_ffn_preserves_whole_math_across_leaf_partitions_and_replay() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let kernels = CudaNativeBlockKernels::load(&ctx).unwrap();
    let mmq = StreamMmq::load_residual2(&ctx).unwrap();
    let module = ctx
        .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
        .unwrap();
    let silu = module.load_function(SILU_MUL_FUNCTION_NAME).unwrap();
    let (hidden, inter) = (512_usize, 768_usize);
    for down_format in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let formats = [GgufBlockFormat::Q4K, GgufBlockFormat::Q4K, down_format];
        let dimensions = [(inter, hidden), (inter, hidden), (hidden, inter)];
        let weights_gpu = formats
            .iter()
            .zip(dimensions)
            .enumerate()
            .map(|(i, (&f, (n, k)))| {
                let (mut bytes, _) = matrix(MatrixFormat::Block(f), n, k, i);
                // Keep this layout/replay fixture finite through the complete
                // FFN while retaining original codes, signs and both formats.
                for block in bytes.chunks_exact_mut(f.block_bytes()) {
                    let offset = if f == GgufBlockFormat::Q6K { 208 } else { 0 };
                    let scale = f16::from_le_bytes([block[offset], block[offset + 1]]);
                    block[offset..offset + 2]
                        .copy_from_slice(&f16::from_f32(scale.to_f32() / 64.0).to_le_bytes());
                    if f == GgufBlockFormat::Q4K {
                        let minimum = f16::from_le_bytes([block[2], block[3]]);
                        block[2..4]
                            .copy_from_slice(&f16::from_f32(minimum.to_f32() / 64.0).to_le_bytes());
                    }
                }
                Guarded::new(&stream, &bytes, 0xab_u8)
            })
            .collect::<Vec<_>>();
        let weights = weights_gpu
            .iter()
            .map(|w| w.pointer(&stream))
            .collect::<Vec<_>>();
        let part = |i: usize, offset| MatrixPart {
            component_id: WeightId::new(format!("fixture.residual2.{i}")).unwrap(),
            format: MatrixFormat::Block(formats[i]),
            rows: dimensions[i].0 as u32,
            columns: dimensions[i].1 as u32,
            output_offset: offset,
            transform: None,
            signs_region: None,
        };
        let gate = [part(0, 0), part(1, inter as u32)];
        let down = [part(2, 0)];
        // This is the production estimator's fixed region: Q4 down may require
        // more K-pack metadata than gate/up. Reuse it between all participants.
        let gate_workspace = mmq
            .workspace(hidden as u32, inter as u32)
            .unwrap()
            .total_bytes;
        let fixed = if down_format == GgufBlockFormat::Q4K {
            gate_workspace.max(
                mmq.workspace(inter as u32, hidden as u32)
                    .unwrap()
                    .total_bytes,
            )
        } else {
            gate_workspace
        };
        for whole in 1..=9_usize {
            let sentinel = f16::from_f32(-12345.0);
            let make_input = |generation: usize| {
                let mut x = vec![sentinel; 8];
                x.extend((0..whole * hidden).map(|i| {
                    f16::from_f32((((i * 13 + generation * 3) % 7) as f32 - 3.0) / 16384.0)
                }));
                x.extend([sentinel; 8]);
                x
            };
            let mut input = stream.clone_htod(&make_input(0)).unwrap();
            let xp = input.device_ptr(&stream).0 + 16;
            // Whole packed, every participant leaf1, and mixed [1,rest].
            let partitions = [
                vec![whole],
                vec![1; whole],
                if whole > 1 {
                    vec![1, whole - 1]
                } else {
                    vec![1]
                },
            ];
            let mut outputs_by_partition = Vec::new();
            for leaves in partitions {
                let packed = leaves.len() == 1;
                let selected = select(
                    &gate,
                    &down,
                    hidden as u64,
                    inter as u64,
                    whole as u64,
                    leaves.len() as u32,
                    packed,
                    packed,
                    Arithmetic::Residual2M2To8,
                )
                .unwrap();
                let residual = selected.mmq_hit.then_some(&mmq);
                assert_eq!(residual.is_some(), (2..=8).contains(&whole));
                assert_eq!(
                    selected.mmq_down_hit,
                    residual.is_some() && down_format == GgufBlockFormat::Q4K
                );
                let layout = ScratchLayout::new(whole as u64, inter as u64).unwrap();
                let scratch = Guarded::new(&stream, &vec![0xcd_u8; fixed as usize], 0xe7);
                let gates = Guarded::new(&stream, &vec![sentinel; whole * inter * 2], sentinel);
                let activation = Guarded::new(&stream, &vec![sentinel; whole * inter], sentinel);
                let output = Guarded::new(&stream, &vec![sentinel; whole * hidden], sentinel);
                assert_eq!(layout.total_bytes as usize, whole * inter * 6);
                let sp = scratch.pointer(&stream);
                let gp = gates.pointer(&stream);
                let ap = activation.pointer(&stream);
                let yp = output.pointer(&stream);
                let launch = || {
                    let mut start = 0_usize;
                    for &count in &leaves {
                        launch_with_policy(
                            &kernels,
                            &silu,
                            &stream,
                            &gate,
                            &down,
                            &weights,
                            xp + (start * hidden * 2) as u64,
                            yp + (start * hidden * 2) as u64,
                            gp + (start * inter * 4) as u64,
                            ap + (start * inter * 2) as u64,
                            count as u32,
                            hidden as u32,
                            inter as u32,
                            0,
                            None,
                            0,
                            residual,
                            sp,
                        )
                        .unwrap();
                        start += count;
                    }
                    assert_eq!(start, whole);
                };
                // Resolve lazy driver/module work before graph capture.
                launch();
                stream.synchronize().unwrap();
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                    .unwrap();
                launch();
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                ).unwrap().expect("complete FFN graph");
                let mut generations = Vec::new();
                for generation in 0..2 {
                    let x = make_input(generation);
                    stream.memcpy_htod(&x, &mut input).unwrap();
                    graph.launch().unwrap();
                    let first = output.read(&stream);
                    assert!(first.iter().all(|x| x.is_finite()));
                    let stage_gate = gates.read(&stream);
                    let stage_activation = activation.read(&stream);
                    assert!(stage_gate
                        .iter()
                        .chain(&stage_activation)
                        .all(|x| x.is_finite()));
                    for (name, values, columns) in [
                        ("activation", stage_activation.as_slice(), inter),
                        ("output", first.as_slice(), hidden),
                    ] {
                        assert!(
                            values
                                .chunks_exact(columns)
                                .all(|row| row.iter().any(|x| x.to_f32() != 0.0)),
                            "{name} row degenerated to zero at M{whole}, generation {generation}"
                        );
                    }
                    scratch.read(&stream);
                    assert_eq!(
                        stream.clone_dtoh(&input).unwrap(),
                        x,
                        "input/guards changed"
                    );
                    graph.launch().unwrap();
                    assert_eq!(output.read(&stream), first, "same input replay changed");
                    generations.push((stage_gate, stage_activation, first));
                }
                assert_ne!(
                    generations[0].0, generations[1].0,
                    "graph reused stale activation pack"
                );
                assert_ne!(
                    generations[0].1, generations[1].1,
                    "changed input did not reach the SiLU activation"
                );
                assert_ne!(
                    generations[0].2, generations[1].2,
                    "changed input did not reach the final output"
                );
                outputs_by_partition.push(generations);
                for w in &weights_gpu {
                    w.assert_unchanged(&stream);
                }
            }
            for partition in &outputs_by_partition[1..] {
                assert_eq!(
                    &outputs_by_partition[0], partition,
                    "whole M{whole} changed arithmetic under leaf subdivision"
                );
            }
        }
    }
}
