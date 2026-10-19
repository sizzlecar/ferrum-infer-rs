//! Actual legacy CUDA instance ownership: graph keys and growing KV storage.
#![cfg(feature = "cuda")]

use ferrum_kernels::backend::{
    cuda::{CudaBackend, CudaState},
    Backend, BackendCompletion, BackendGraph, CudaBuf, Dtype, KvCache, KvFp16, KvLayer,
};
use half::f16;

fn complete(context: &mut CudaState) {
    assert!(matches!(
        CudaBackend::observe_completion(context),
        BackendCompletion::Succeeded
    ));
}

struct Projection {
    input: CudaBuf,
    bias: CudaBuf,
    output: CudaBuf,
    expected: Vec<f32>,
}

impl Projection {
    fn new(context: &mut CudaState, seed: f32) -> Self {
        // Cross a launch block boundary and keep every operation exact in F32.
        let input: Vec<_> = (0..257).map(|i| seed + (i % 17) as f32 * 0.25).collect();
        let bias: Vec<_> = (0..257).map(|i| ((i % 5) as f32 - 2.0) * 0.125).collect();
        Self {
            input: CudaBackend::from_slice_typed_in_context(context, &input),
            bias: CudaBackend::from_slice_typed_in_context(context, &bias),
            output: CudaBackend::alloc_typed_in_context(context, Dtype::F32, input.len()),
            expected: input.iter().zip(bias).map(|(a, b)| a + b).collect(),
        }
    }

    fn encode(&mut self, context: &mut CudaState) {
        CudaBackend::copy_slice(
            context,
            &self.input,
            0,
            &mut self.output,
            0,
            self.expected.len(),
        );
        CudaBackend::add_inplace(context, &mut self.output, &self.bias, self.expected.len());
    }

    fn capture(&mut self, context: &mut CudaState, key: u64) {
        // Resolve the actual PTX module before capture, without a fake provider.
        CudaBackend::begin_execution(context).unwrap();
        self.encode(context);
        complete(context);
        CudaBackend::begin_execution(context).unwrap();
        CudaBackend::begin_graph_capture(context).unwrap();
        self.encode(context);
        CudaBackend::end_graph_capture(context, key).unwrap();
        complete(context);
    }

    fn replay_and_check(&mut self, context: &mut CudaState, key: u64) {
        CudaBackend::begin_execution(context).unwrap();
        CudaBackend::write_typed(
            context,
            &mut self.output,
            &vec![f32::NAN; self.expected.len()],
        );
        let hit = CudaBackend::replay_graph(context, key).unwrap();
        complete(context);
        assert!(hit, "the original captured executable must remain cached");
        let actual = CudaBackend::to_vec_in_context(context, &self.output, self.expected.len());
        assert_eq!(
            actual, self.expected,
            "all output elements must belong to this instance"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device and actual graph capture/replay"]
fn equal_graph_keys_remain_owned_across_peer_reset_and_retirement() {
    const KEY: u64 = 8;
    let mut first = CudaBackend::new_context();
    CudaBackend::begin_execution(&mut first).unwrap();
    let mut a = Projection::new(&mut first, 2.0);
    complete(&mut first);
    a.capture(&mut first, KEY);

    // Construct B only after A has installed its real executable.
    let mut second = CudaBackend::new_context();
    assert_eq!(first.ordinal, second.ordinal);
    assert!(!first.stream.cu_stream().is_null());
    assert!(!second.stream.cu_stream().is_null());
    assert_ne!(first.stream.cu_stream(), second.stream.cu_stream());
    CudaBackend::begin_execution(&mut second).unwrap();
    let mut b = Projection::new(&mut second, -5.0);
    complete(&mut second);
    b.capture(&mut second, KEY);
    assert_ne!(a.expected, b.expected);
    a.replay_and_check(&mut first, KEY);
    b.replay_and_check(&mut second, KEY);

    CudaBackend::begin_execution(&mut second).unwrap();
    CudaBackend::reset_graph(&mut second, KEY);
    let removed = !CudaBackend::replay_graph(&mut second, KEY).unwrap();
    complete(&mut second);
    assert!(removed);
    a.replay_and_check(&mut first, KEY);

    b.capture(&mut second, KEY);
    CudaBackend::begin_execution(&mut second).unwrap();
    CudaBackend::reset_all_graphs(&mut second);
    let removed = !CudaBackend::replay_graph(&mut second, KEY).unwrap();
    complete(&mut second);
    assert!(removed);
    a.replay_and_check(&mut first, KEY);

    // Retire B with a populated cache; A and its captured buffers remain live.
    b.capture(&mut second, KEY);
    b.replay_and_check(&mut second, KEY);
    CudaBackend::retire_context(&mut second).unwrap();
    drop(b);
    drop(second);
    a.replay_and_check(&mut first, KEY);
    CudaBackend::retire_context(&mut first).unwrap();
    drop(a);
    drop(first);
}

const HEADS: usize = 3;
const HEAD_DIM: usize = 5;

fn cache_values(seed: f32, tokens: usize) -> Vec<f32> {
    (0..HEADS)
        .flat_map(|head| {
            (0..tokens).flat_map(move |token| {
                (0..HEAD_DIM).map(move |dimension| {
                    seed + head as f32 * 4.0 + token as f32 * 0.5 + dimension as f32 * 0.125
                })
            })
        })
        .collect()
}

fn initialize_cache(context: &mut CudaState, capacity: usize, seed: f32) -> KvCache<CudaBackend> {
    let mut cache =
        <KvFp16 as KvLayer<CudaBackend>>::alloc_contig(context, capacity, HEADS, HEAD_DIM);
    for (buffer, seed) in [(&mut cache.k, seed), (&mut cache.v, -seed)] {
        let values: Vec<_> = cache_values(seed, capacity)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        CudaBackend::write_typed(context, buffer, &values);
    }
    <KvFp16 as KvLayer<CudaBackend>>::set_len(&mut cache, capacity);
    cache
}

fn check_cache(context: &mut CudaState, cache: &KvCache<CudaBackend>, seed: f32) {
    assert_eq!((cache.num_kv_heads, cache.head_dim), (HEADS, HEAD_DIM));
    assert!(cache.len <= cache.capacity);
    for (buffer, seed) in [(&cache.k, seed), (&cache.v, -seed)] {
        CudaBackend::begin_execution(context).unwrap();
        let valid_per_head = cache.len * HEAD_DIM;
        let mut packed = CudaBackend::alloc_in_context(context, HEADS * valid_per_head);
        // The new capacity is uninitialized outside the preserved prefix.
        // Read every valid element using the new physical head stride only.
        for head in 0..HEADS {
            CudaBackend::copy_slice(
                context,
                buffer,
                head * cache.capacity * HEAD_DIM,
                &mut packed,
                head * valid_per_head,
                valid_per_head,
            );
        }
        complete(context);
        let actual = CudaBackend::to_vec_in_context(context, &packed, HEADS * valid_per_head);
        assert_eq!(actual, cache_values(seed, cache.len));
    }
}

#[test]
#[ignore = "requires a real CUDA device and asynchronous multi-head KV growth"]
fn contiguous_kv_growth_keeps_each_instances_original_rows() {
    let mut first = CudaBackend::new_context();
    let mut second = CudaBackend::new_context();
    assert_eq!(first.ordinal, second.ordinal);
    assert_ne!(first.stream.cu_stream(), second.stream.cu_stream());
    CudaBackend::begin_execution(&mut first).unwrap();
    let mut a = initialize_cache(&mut first, 2, 1.0);
    CudaBackend::begin_execution(&mut second).unwrap();
    let mut b = initialize_cache(&mut second, 3, -3.0);
    complete(&mut first);
    complete(&mut second);

    // Both streams grow actual buffers before either completion is observed.
    // Head stride changes from old capacity to new capacity in every copy.
    CudaBackend::begin_execution(&mut first).unwrap();
    <KvFp16 as KvLayer<CudaBackend>>::ensure_contig_capacity(
        &mut first, &mut a, 5, HEADS, HEAD_DIM,
    )
    .unwrap();
    CudaBackend::begin_execution(&mut second).unwrap();
    <KvFp16 as KvLayer<CudaBackend>>::ensure_contig_capacity(
        &mut second,
        &mut b,
        6,
        HEADS,
        HEAD_DIM,
    )
    .unwrap();
    complete(&mut first);
    complete(&mut second);
    assert_eq!((a.capacity, a.len), (5, 2));
    assert_eq!((b.capacity, b.len), (6, 3));
    check_cache(&mut first, &a, 1.0);
    check_cache(&mut second, &b, -3.0);

    CudaBackend::retire_context(&mut second).unwrap();
    drop(b);
    drop(second);
    CudaBackend::begin_execution(&mut first).unwrap();
    <KvFp16 as KvLayer<CudaBackend>>::ensure_contig_capacity(
        &mut first, &mut a, 7, HEADS, HEAD_DIM,
    )
    .unwrap();
    complete(&mut first);
    assert_eq!((a.capacity, a.len), (7, 2));
    check_cache(&mut first, &a, 1.0);
    CudaBackend::retire_context(&mut first).unwrap();
    drop(a);
    drop(first);
}
