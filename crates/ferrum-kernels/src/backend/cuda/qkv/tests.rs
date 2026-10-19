use super::*;
use crate::backend::{BackendCompletion, BackendGraph};
use cudarc::driver::DevicePtr;
use half::f16;

const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = 64;
const TOKENS: usize = 3;
const CANARY: usize = 7;
const EPS: f32 = 1.0e-5;

struct Fixture {
    // The context retires graphs before the referenced external allocations.
    ctx: CudaState,
    input: CudaBuf,
    q_weight: CudaBuf,
    k_weight: CudaBuf,
    cos: CudaBuf,
    sin: CudaBuf,
    q: CudaBuf,
    k: CudaBuf,
    v: CudaBuf,
    host_input: Vec<f32>,
    host_weight: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
        let mut ctx =
            CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        let host_input: Vec<_> = (0..TOKENS * (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM)
            .map(|i| ((i % 43) as f32 - 21.0) / 16.0)
            .collect();
        let host_weight: Vec<_> = (0..HEAD_DIM).map(|i| 1.0 + (i % 4) as f32 / 8.0).collect();
        let cos: Vec<_> = [1.0, 0.0, -1.0, 0.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, HEAD_DIM / 2))
            .collect();
        let sin: Vec<_> = [0.0, 1.0, 0.0, -1.0]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, HEAD_DIM / 2))
            .collect();
        let input = CudaBackend::from_slice_in_context(&mut ctx, &host_input);
        let q_weight = CudaBackend::from_slice_in_context(&mut ctx, &host_weight);
        let k_weight = CudaBackend::from_slice_in_context(&mut ctx, &host_weight);
        let cos = CudaBackend::from_slice_in_context(&mut ctx, &cos);
        let sin = CudaBackend::from_slice_in_context(&mut ctx, &sin);
        let q = CudaBackend::from_slice_in_context(
            &mut ctx,
            &vec![-7.0; TOKENS * Q_HEADS * HEAD_DIM + CANARY],
        );
        let k = CudaBackend::from_slice_in_context(
            &mut ctx,
            &vec![-7.0; TOKENS * KV_HEADS * HEAD_DIM + CANARY],
        );
        let v = CudaBackend::from_slice_in_context(
            &mut ctx,
            &vec![-7.0; TOKENS * KV_HEADS * HEAD_DIM + CANARY],
        );
        Self {
            ctx,
            input,
            q_weight,
            k_weight,
            cos,
            sin,
            q,
            k,
            v,
            host_input,
            host_weight,
        }
    }

    fn encode(&mut self, tokens: usize) {
        CudaBackend::split_qkv_norm_rope(
            &mut self.ctx,
            &self.input,
            &self.q_weight,
            &self.k_weight,
            &self.cos,
            &self.sin,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            tokens,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            1,
            EPS,
            1,
        )
        .unwrap();
    }

    fn pointers(&self) -> [u64; 3] {
        [
            &self.ctx.qkv_scratch.q,
            &self.ctx.qkv_scratch.k,
            &self.ctx.qkv_scratch.v,
        ]
        .map(|slot| {
            slot.as_ref()
                .unwrap()
                .as_f16()
                .device_ptr(&self.ctx.stream)
                .0
        })
    }

    fn check(&mut self, tokens: usize, position: usize) {
        CudaBackend::observe_completion(&mut self.ctx)
            .into_result()
            .unwrap();
        let stride = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM;
        for (kind, heads, offset, buffer) in [
            (0, Q_HEADS, 0, &self.q),
            (1, KV_HEADS, Q_HEADS * HEAD_DIM, &self.k),
            (2, KV_HEADS, (Q_HEADS + KV_HEADS) * HEAD_DIM, &self.v),
        ] {
            let actual = CudaBackend::to_vec_in_context(&mut self.ctx, buffer, buffer.len());
            for head in 0..heads {
                for token in 0..tokens {
                    let start = token * stride + offset + head * HEAD_DIM;
                    let row = &self.host_input[start..start + HEAD_DIM];
                    let scale = 1.0
                        / (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>()
                            / HEAD_DIM as f64
                            + f64::from(EPS))
                        .sqrt();
                    for dim in 0..HEAD_DIM {
                        let expected = if kind == 2 {
                            row[dim]
                        } else {
                            let half = HEAD_DIM / 2;
                            let a = dim % half;
                            let x0 = f64::from(row[a]) * scale * f64::from(self.host_weight[a]);
                            let x1 = f64::from(row[a + half])
                                * scale
                                * f64::from(self.host_weight[a + half]);
                            let (cos, sin) = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
                                [position + token];
                            f16::from_f64(if dim < half {
                                x0 * cos - x1 * sin
                            } else {
                                x1 * cos + x0 * sin
                            })
                            .to_f32()
                        };
                        let value = actual[(head * tokens + token) * HEAD_DIM + dim];
                        assert!((value - expected).abs() <= 0.003, "kind={kind} head={head} token={token} dim={dim}: {value} != {expected}");
                    }
                }
            }
            assert_eq!(&actual[TOKENS * heads * HEAD_DIM..], &[-7.0; CANARY]);
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device and graph replay"]
fn qkv_capture_reuses_original_scratch_and_replays_after_eager_growth() {
    let mut fixture = Fixture::new();
    fixture.encode(1);
    fixture.check(1, 1);
    let original = fixture.pointers();
    CudaBackend::set_decode_state(&mut fixture.ctx, 0, 1);
    CudaBackend::set_dev_state_mode(&mut fixture.ctx, true);
    unsafe { CudaBackend::begin_graph_capture(&mut fixture.ctx) }.unwrap();
    fixture.encode(1);
    assert_eq!(fixture.pointers(), original);
    unsafe { CudaBackend::end_graph_capture(&mut fixture.ctx, 71) }.unwrap();
    assert!(unsafe { CudaBackend::replay_graph(&mut fixture.ctx, 71) }.unwrap());
    fixture.check(1, 1);

    // Growing eager scratch cannot release the addresses captured by key 71.
    fixture.encode(TOKENS);
    fixture.check(TOKENS, 1);
    assert_ne!(fixture.pointers(), original);
    assert_eq!(fixture.ctx.captured_storage.len(), 3);
    CudaBackend::set_decode_state(&mut fixture.ctx, 0, 2);
    assert!(unsafe { CudaBackend::replay_graph(&mut fixture.ctx, 71) }.unwrap());
    fixture.check(1, 2);
    assert_eq!(fixture.ctx.captured_storage.len(), 3);
    CudaBackend::reset_graph(&mut fixture.ctx, 71);
    CudaBackend::observe_completion(&mut fixture.ctx)
        .into_result()
        .unwrap();
    assert!(fixture.ctx.captured_storage.is_empty());
    CudaBackend::retire_context(&mut fixture.ctx).unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device and failed capture cancellation"]
fn qkv_unprepared_capture_failure_discards_prefix_and_reuses_original_context() {
    let mut fixture = Fixture::new();
    // Warm only the zero kernel. QKV scratch intentionally remains absent.
    CudaBackend::zero_buffer(&mut fixture.ctx, &mut fixture.q, 1).unwrap();
    CudaBackend::write_f32_to_activation(&mut fixture.ctx, &mut fixture.q, &[-7.0]);
    CudaBackend::observe_completion(&mut fixture.ctx)
        .into_result()
        .unwrap();
    unsafe { CudaBackend::begin_graph_capture(&mut fixture.ctx) }.unwrap();
    CudaBackend::zero_buffer(&mut fixture.ctx, &mut fixture.q, Q_HEADS * HEAD_DIM).unwrap();
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fixture.encode(1)));
    assert!(failure.is_err());
    // Ordinary observation must still preserve a live capture, never cancel it.
    assert!(matches!(
        CudaBackend::observe_completion(&mut fixture.ctx),
        BackendCompletion::Unresolved(_)
    ));
    CudaBackend::abort_graph_capture(&mut fixture.ctx).unwrap();
    CudaBackend::observe_completion(&mut fixture.ctx)
        .into_result()
        .unwrap();
    assert_eq!(
        CudaBackend::to_vec_in_context(&mut fixture.ctx, &fixture.q, fixture.q.len()),
        vec![-7.0; fixture.q.len()]
    );
    assert!(fixture.ctx.graph_cache.is_empty());
    assert!(fixture.ctx.graph_leases.is_empty());
    CudaBackend::begin_execution(&mut fixture.ctx).unwrap();
    fixture.encode(1);
    fixture.check(1, 1);
    CudaBackend::retire_context(&mut fixture.ctx).unwrap();
}

#[test]
#[ignore = "requires an actual CUDA device and driver-invalidated capture"]
fn invalidated_capture_is_confirmed_ended_before_original_context_reuse() {
    use cudarc::driver::sys;
    let mut fixture = Fixture::new();
    fixture.encode(1);
    fixture.check(1, 1);
    unsafe { CudaBackend::begin_graph_capture(&mut fixture.ctx) }.unwrap();
    fixture.encode(1);
    // Synchronizing a capturing stream is a real prohibited CUDA operation,
    // invalidating this capture. Do not fabricate the driver error/status.
    let sync = unsafe { sys::cuStreamSynchronize(fixture.ctx.stream.cu_stream()) };
    let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let query = unsafe { sys::cuStreamIsCapturing(fixture.ctx.stream.cu_stream(), &mut status) };
    // Finish cleanup before asserting the driver fault, so a changed error
    // code cannot leave this test's real capture active during unwinding.
    let cancellation = CudaBackend::abort_graph_capture(&mut fixture.ctx);
    let completion = CudaBackend::observe_completion(&mut fixture.ctx);
    assert_eq!(sync, sys::CUresult::CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    assert_eq!(query, sys::CUresult::CUDA_SUCCESS);
    assert_eq!(
        status,
        sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED
    );
    assert!(cancellation.is_err());
    assert!(!fixture.ctx.capture_in_flight);
    completion.into_result().unwrap();
    CudaBackend::begin_execution(&mut fixture.ctx).unwrap();
    fixture.encode(1);
    fixture.check(1, 1);
    CudaBackend::retire_context(&mut fixture.ctx).unwrap();
}
