//! Backend oracle for the two production BatchDecode enqueue paths. This does
//! not construct a Core admission or qualify complete model output.
use super::*;
use crate::backend::cuda::vnext_ops::{
    cuda_vnext_runtime_config, transformer::test_support::Guarded,
};
use cudarc::driver::{
    sys::{CUgraphInstantiate_flags, CUstreamCaptureMode},
    CudaSlice, DevicePtr,
};
use ferrum_interfaces::vnext::DeviceId;
use half::f16;

const GUARD_WORDS: usize = 8;

#[cfg(feature = "cuda-indexed-causal-attention")]
#[path = "fixed_replay_gpu.rs"]
mod fixed_replay_gpu;

struct PhysicalPage {
    allocation: Arc<Guarded<f16>>,
    element_offset: usize,
}

impl PhysicalPage {
    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.allocation.pointer(stream) + (self.element_offset * 2) as u64
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<f16> {
        let whole_allocation = self.allocation.read(stream);
        whole_allocation
            [self.element_offset..self.element_offset + VNEXT_KV_PAGE_BYTES as usize / 2]
            .to_vec()
    }
}

struct RequestPages {
    pages: Vec<PhysicalPage>,
    previous: Vec<Vec<u16>>,
    seed: usize,
}

impl RequestPages {
    fn grow_to(&mut self, stream: &Arc<CudaStream>, shape: CausalAttentionShape, length: u64) {
        let count = (shape.physical_state_bytes(length).unwrap() / VNEXT_KV_PAGE_BYTES) as usize;
        while self.pages.len() < count {
            let page = self.pages.len();
            let values = (0..VNEXT_KV_PAGE_BYTES as usize / 2)
                .map(|i| {
                    f16::from_f32(
                        (((i * 7 + page * 17 + self.seed * 29) % 127) as f32 - 63.0) / 512.0,
                    )
                })
                .collect::<Vec<_>>();
            self.previous
                .push(values.iter().map(|value| value.to_bits()).collect());
            self.pages.push(PhysicalPage {
                allocation: Arc::new(Guarded::new(stream, &values, f16::from_f32(71.0))),
                element_offset: 0,
            });
        }
    }
}

struct Arm {
    scratch: CudaSlice<f16>,
    binding: CudaSlice<u64>,
    requests: Vec<RequestPages>,
}

impl Arm {
    fn scratch_pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.scratch.device_ptr(stream).0 + (GUARD_WORDS * 2) as u64
    }

    fn binding_pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        self.binding.device_ptr(stream).0 + (GUARD_WORDS * 8) as u64
    }

    fn binding_words(
        &self,
        stream: &Arc<CudaStream>,
        shape: CausalAttentionShape,
        layout: BindingLayout,
        requests: &[usize],
        lengths: &[u64],
    ) -> Vec<u64> {
        // This raw-backend fixture supplies the private independence input
        // only after checking its real page ranges. It does not mint or stand
        // in for the production Core Sequence-view proof.
        let mut ranges = requests
            .iter()
            .flat_map(|&request| {
                self.requests[request].pages.iter().map(|page| {
                    let start = page.pointer(stream);
                    (start, start.checked_add(VNEXT_KV_PAGE_BYTES).unwrap())
                })
            })
            .collect::<Vec<_>>();
        ranges.sort_unstable();
        assert!(ranges.windows(2).all(|pair| pair[0].1 <= pair[1].0));
        let mut words = vec![u64::MAX; layout.required_bytes as usize / 8 + 2 * GUARD_WORDS];
        for (slot, (&request, &length)) in requests.iter().zip(lengths).enumerate() {
            let pages = self.requests[request]
                .pages
                .iter()
                .map(|page| page.pointer(stream))
                .collect::<Vec<_>>();
            let entries = shape.table_entries(length).unwrap() as i32;
            let addresses = binding_addresses(shape.kv_layout().unwrap(), entries, &pages).unwrap();
            let base = GUARD_WORDS + layout.binding_offset(slot).unwrap() as usize / 8;
            // Same six-i32 control ABI as binding_payload. Unused slot tails
            // remain poison so a cached count cannot silently read live pages.
            words[base] = entries as u32 as u64 | ((length - 1) << 32);
            words[base + 1] = 1 | (length << 32);
            words[base + 2] = slot as u64;
            words[base + 3..base + 3 + addresses.len()].copy_from_slice(&addresses);
        }
        words
    }

    fn share_allocation_with_disjoint_first_pages(
        &mut self,
        stream: &Arc<CudaStream>,
        first: usize,
    ) {
        let values = self.requests[first].previous[0]
            .iter()
            .chain(&self.requests[first + 1].previous[0])
            .copied()
            .map(f16::from_bits)
            .collect::<Vec<_>>();
        let allocation = Arc::new(Guarded::new(stream, &values, f16::from_f32(72.0)));
        self.requests[first].pages[0] = PhysicalPage {
            allocation: Arc::clone(&allocation),
            element_offset: 0,
        };
        self.requests[first + 1].pages[0] = PhysicalPage {
            allocation,
            element_offset: VNEXT_KV_PAGE_BYTES as usize / 2,
        };
        let a = &self.requests[first].pages[0];
        let b = &self.requests[first + 1].pages[0];
        assert!(Arc::ptr_eq(&a.allocation, &b.allocation));
        assert_eq!(a.pointer(stream) + VNEXT_KV_PAGE_BYTES, b.pointer(stream));
    }
}

fn scratch_words(
    shape: CausalAttentionShape,
    layout: ScratchLayout,
    requests: &[usize],
    wave: usize,
) -> Vec<f16> {
    let mut values = vec![f16::ZERO; layout.required_bytes as usize / 2 + 2 * GUARD_WORDS];
    values[..GUARD_WORDS].fill(f16::from_f32(73.0));
    let end = values.len() - GUARD_WORDS;
    values[end..].fill(f16::from_f32(73.0));
    for (slot, request) in requests.iter().enumerate() {
        for (kind, base, width) in [
            (0, layout.query_raw, shape.query_projection_features),
            (1, layout.key_raw, shape.kv_features),
            (2, layout.value_raw, shape.kv_features),
        ] {
            let begin =
                GUARD_WORDS + layout.token_offset(base, slot as u64, width).unwrap() as usize / 2;
            for (i, value) in values[begin..begin + width as usize].iter_mut().enumerate() {
                *value = f16::from_f32(
                    (((i * 11 + request * 19 + wave * 23 + kind * 31) % 97) as f32 - 48.0) / 128.0,
                );
            }
        }
    }
    values
}

fn bits(values: &[f16]) -> Vec<u16> {
    values.iter().map(|value| value.to_bits()).collect()
}

// Exact writable elements of the current token in the native VLLM block
// layout. All other physical page elements, including unused capacity, must
// retain their preceding-wave bytes independently of cross-arm equality.
fn writable_elements(shape: CausalAttentionShape, position: u64) -> (usize, Vec<bool>) {
    let CausalKvLayout::VllmBlocks16 {
        combined_block_bytes,
        blocks_per_page,
    } = shape.kv_layout().unwrap()
    else {
        panic!("fixture must use the addressed VLLM physical layout");
    };
    let block = position / VLLM_BLOCK_TOKENS;
    let page = block / blocks_per_page;
    let block_base = (block % blocks_per_page * combined_block_bytes / 2) as usize;
    let token = (position % VLLM_BLOCK_TOKENS) as usize;
    let dim = shape.head_dim as usize;
    let heads = shape.key_value_heads as usize;
    let mut mask = vec![false; VNEXT_KV_PAGE_BYTES as usize / 2];
    for head in 0..heads {
        for d in 0..dim {
            mask[block_base + head * dim * 16 + (d / 8) * 16 * 8 + token * 8 + d % 8] = true;
            mask[block_base + heads * dim * 16 + head * dim * 16 + d * 16 + token] = true;
        }
    }
    (page as usize, mask)
}

fn check_pages(
    arms: &mut [Arm; 2],
    stream: &Arc<CudaStream>,
    shape: CausalAttentionShape,
    requests: &[usize],
    lengths: &[u64],
) {
    let [reference, candidate] = arms;
    for (request, (left, right)) in reference
        .requests
        .iter_mut()
        .zip(&mut candidate.requests)
        .enumerate()
    {
        let active = requests
            .iter()
            .position(|&current| current == request)
            .map(|slot| writable_elements(shape, lengths[slot] - 1));
        let mut changed = false;
        for (page, (a, b)) in left.pages.iter().zip(&right.pages).enumerate() {
            let actual = bits(&a.read(stream));
            assert_eq!(
                actual,
                bits(&b.read(stream)),
                "complete physical KV differs"
            );
            for (i, (&now, &before)) in actual.iter().zip(&left.previous[page]).enumerate() {
                if active
                    .as_ref()
                    .is_some_and(|(p, mask)| *p == page && mask[i])
                {
                    changed |= now != before;
                } else {
                    assert_eq!(
                        now, before,
                        "write escaped current token's physical KV range"
                    );
                }
            }
            left.previous[page] = actual.clone();
            right.previous[page] = actual;
        }
        if active.is_some() {
            assert!(
                changed,
                "current binding did not reach this request's KV allocation"
            );
        }
    }
}

#[test]
#[ignore = "requires actual CUDA and the pinned addressed-vLLM operator set"]
fn packed_prepare_gate_matches_loop_across_current_bindings_pages_and_requests() {
    compare_batches(false);
}

#[test]
#[cfg(feature = "cuda-indexed-causal-attention")]
#[ignore = "requires actual CUDA and the pinned indexed addressed-vLLM operator set"]
fn indexed_groups_match_contiguous_across_growth_reordering_and_path_transition() {
    compare_batches(true);
}

fn compare_batches(indexed: bool) {
    let runtime = CudaDeviceRuntime::new(
        cuda_vnext_runtime_config(
            0,
            DeviceId::new("device.test.packed-decode-prepare-gate").unwrap(),
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap(),
    )
    .unwrap();
    let provider =
        CudaCausalPagedAttentionProvider::new(&runtime, AttentionExecutionPolicy::NativeAdaptive)
            .unwrap();
    let stream = runtime.context().new_stream().unwrap();
    for (head_dim, participants, output_gate) in [(128, 8, false), (256, 32, true)] {
        let mut shape = shape(head_dim);
        shape.output_gate = output_gate;
        shape.query_projection_features = shape.query_features * if output_gate { 2 } else { 1 };
        let cuda_shape = shape.cuda_shape().unwrap();
        let binding_layout = BindingLayout::new(shape, participants).unwrap();
        let layout = ScratchLayout::for_participants(
            shape,
            participants as u64,
            participants,
            CausalProjection::F16,
            AttentionExecutionPolicy::NativeAdaptive,
        )
        .unwrap();
        let page_tokens = VNEXT_KV_PAGE_BYTES / shape.state_bytes_per_token().unwrap();
        let initial_lengths = (0..participants)
            .map(|slot| {
                if slot % 4 < 2 {
                    page_tokens
                } else {
                    1024 + page_tokens
                }
            })
            .collect::<Vec<_>>();
        for &length in &initial_lengths {
            assert_eq!(
                shape.physical_state_bytes(length + 1).unwrap(),
                shape.physical_state_bytes(length).unwrap() + VNEXT_KV_PAGE_BYTES
            );
        }
        let launches = build_launches(shape, binding_layout, layout, &initial_lengths);
        assert!(launches
            .iter()
            .any(|x| x.path == CausalAttentionKernelPath::VllmAddressedDecodeV1));
        assert!(launches
            .iter()
            .any(|x| x.path == CausalAttentionKernelPath::VllmAddressedDecodeV2));
        let batch_modes = if indexed {
            [
                (true, CausalDecodeAttentionGrouping::Contiguous),
                (true, CausalDecodeAttentionGrouping::Indexed),
            ]
        } else {
            [
                (false, CausalDecodeAttentionGrouping::Contiguous),
                (true, CausalDecodeAttentionGrouping::Contiguous),
            ]
        };
        let batches = batch_modes.map(|(packed, grouping)| {
            BatchDecode::for_launches(
                &launches,
                true,
                binding_layout,
                shape,
                layout,
                packed,
                true, // Independently owned, nonoverlapping fixture KV ranges.
                grouping,
            )
            .unwrap()
            .unwrap()
        });
        assert_eq!(batches[0].packed_work().is_some(), indexed);
        if indexed {
            assert!(batches[0].groups.len() > 2);
            assert_eq!(batches[1].groups.len(), 2);
            assert!(batches[1].row_map.is_some());
        }
        assert!(batches[1].packed_work().unwrap().prepare.is_some());
        assert_eq!(
            batches[1].packed_work().unwrap().gate_tokens(),
            Some(participants as u64)
        );
        let query_norm = Guarded::new(
            &stream,
            &(0..head_dim)
                .map(|i| f16::from_f32(0.75 + (i % 11) as f32 / 32.0))
                .collect::<Vec<_>>(),
            f16::from_f32(74.0),
        );
        let key_norm = Guarded::new(
            &stream,
            &vec![f16::ONE; head_dim as usize],
            f16::from_f32(75.0),
        );
        let mut requests = (0..participants).collect::<Vec<_>>();
        let initial_scratch = scratch_words(shape, layout, &requests, 0);
        let mut arms = [false, true].map(|_| Arm {
            scratch: stream.clone_htod(&initial_scratch).unwrap(),
            binding: stream
                .clone_htod(&vec![
                    u64::MAX;
                    binding_layout.required_bytes as usize / 8
                        + 2 * GUARD_WORDS
                ])
                .unwrap(),
            requests: Vec::new(),
        });
        for arm in &mut arms {
            for (seed, &length) in initial_lengths.iter().enumerate() {
                let mut request = RequestPages {
                    pages: Vec::new(),
                    previous: Vec::new(),
                    seed,
                };
                request.grow_to(&stream, shape, length);
                arm.requests.push(request);
            }
            // The first V1 pair writes different physical pages of ONE device
            // allocation. Buffer identity alone is not the independence rule.
            arm.share_allocation_with_disjoint_first_pages(&stream, 0);
        }
        // Capture the complete production prepare -> gather -> selected V1/V2
        // groups -> optional gate path for each arm. Only payloads change later.
        let mut graphs = capture_batches(
            &stream,
            &provider.functions,
            &arms,
            &batches,
            &launches,
            binding_layout,
            cuda_shape,
            layout,
            &query_norm,
            &key_norm,
            output_gate,
        );
        let mut previous_query = None;
        for wave in 0..if indexed { 6 } else { 4 } {
            if wave == 2 {
                // Fresh requests really own new physical allocations. The old
                // owners remain alive and must not change on the new replays.
                for arm in &mut arms {
                    for (index, &length) in initial_lengths.iter().enumerate() {
                        let mut request = RequestPages {
                            pages: Vec::new(),
                            previous: Vec::new(),
                            seed: participants + index,
                        };
                        request.grow_to(&stream, shape, length);
                        arm.requests.push(request);
                    }
                    arm.share_allocation_with_disjoint_first_pages(&stream, participants);
                }
                requests = (0..participants)
                    .map(|slot| participants + (slot ^ 1))
                    .collect();
            }
            let lengths = initial_lengths
                .iter()
                .map(|length| {
                    if indexed && wave >= 4 && *length <= 512 {
                        512 + (wave - 4) as u64
                    } else {
                        length + (wave % 2) as u64
                    }
                })
                .collect::<Vec<_>>();
            let current = build_launches(shape, binding_layout, layout, &lengths);
            if indexed && wave == 5 {
                let updated = batch_modes.map(|(packed, grouping)| {
                    BatchDecode::for_launches(
                        &current,
                        true,
                        binding_layout,
                        shape,
                        layout,
                        packed,
                        true,
                        grouping,
                    )
                    .unwrap()
                    .unwrap()
                });
                let key = |batch: &BatchDecode| {
                    batch
                        .bind_replay(CudaCommandReplayKeyBuilder::new("fixture", "causal"))
                        .finish()
                        .bind_runtime_payload("causal", std::iter::empty(), &[])
                };
                assert_ne!(key(&batches[1]), key(&updated[1]));
                assert_eq!(updated[1].v1_count, 0);
                assert_eq!(updated[1].groups.len(), 1);
                // A length crossing 512 changes the host topology/map and
                // captures a new graph. Existing replay must not choose math.
                stream.synchronize().unwrap();
                graphs.clear();
                graphs = capture_batches(
                    &stream,
                    &provider.functions,
                    &arms,
                    &updated,
                    &current,
                    binding_layout,
                    cuda_shape,
                    layout,
                    &query_norm,
                    &key_norm,
                    output_gate,
                );
            } else {
                for (before, now) in launches.iter().zip(&current) {
                    assert_eq!(before.path, now.path);
                    assert_eq!(before.replay_topology, now.replay_topology);
                }
            }
            let input = scratch_words(shape, layout, &requests, wave);
            let expected_bindings = arms
                .iter_mut()
                .map(|arm| {
                    for (&request, &length) in requests.iter().zip(&lengths) {
                        arm.requests[request].grow_to(&stream, shape, length);
                    }
                    let words =
                        arm.binding_words(&stream, shape, binding_layout, &requests, &lengths);
                    stream.memcpy_htod(&words, &mut arm.binding).unwrap();
                    stream.memcpy_htod(&input, &mut arm.scratch).unwrap();
                    words
                })
                .collect::<Vec<_>>();
            for graph in &graphs {
                graph.launch().unwrap();
            }
            stream.synchronize().unwrap();
            let actual = arms
                .iter()
                .map(|arm| stream.clone_dtoh(&arm.scratch).unwrap())
                .collect::<Vec<_>>();
            if !indexed {
                assert_eq!(
                    bits(&actual[0]),
                    bits(&actual[1]),
                    "whole scratch differs, including prepared query/context and partition output"
                );
            } else {
                for (base, width) in [
                    (layout.query, shape.query_features),
                    (layout.context, shape.query_features),
                ] {
                    let begin = GUARD_WORDS + base as usize / 2;
                    let end = begin + participants * width as usize;
                    assert_eq!(
                        bits(&actual[0][begin..end]),
                        bits(&actual[1][begin..end]),
                        "indexed grouping changed original-row query/context bits"
                    );
                }
                // V2 temporary rows are group-local and intentionally differ.
                // Check actual dense lengths/map plus the untouched padding.
                let partition = layout.vllm.unwrap();
                let read_i32 = |words: &[f16], base: u64| {
                    (0..participants)
                        .map(|i| {
                            let j = GUARD_WORDS + base as usize / 2 + i * 2;
                            (u32::from(words[j].to_bits())
                                | (u32::from(words[j + 1].to_bits()) << 16))
                                as i32
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    read_i32(&actual[1], partition.sequence_lengths),
                    lengths.iter().map(|&x| x as i32).collect::<Vec<_>>()
                );
                let expected_map =
                    indexed_row_map(&current.iter().map(|x| x.path).collect::<Vec<_>>()).unwrap();
                assert_eq!(read_i32(&actual[1], partition.row_indices), expected_map);
                let padding = GUARD_WORDS + (partition.row_indices as usize + participants * 4) / 2;
                for words in &actual {
                    assert_eq!(
                        bits(&words[padding..words.len() - GUARD_WORDS]),
                        bits(&input[padding..input.len() - GUARD_WORDS]),
                        "gather escaped admitted length/map arrays"
                    );
                }
            }
            for (arm, binding) in arms.iter().zip(&expected_bindings) {
                assert_eq!(
                    stream.clone_dtoh(&arm.binding).unwrap(),
                    *binding,
                    "binding/control/slot guards modified"
                );
            }
            for words in &actual {
                assert_eq!(&words[..GUARD_WORDS], &input[..GUARD_WORDS]);
                assert_eq!(
                    &words[words.len() - GUARD_WORDS..],
                    &input[input.len() - GUARD_WORDS..]
                );
                for (base, width) in [
                    (layout.query_raw, shape.query_projection_features),
                    (layout.key_raw, shape.kv_features),
                    (layout.value_raw, shape.kv_features),
                ] {
                    let begin = GUARD_WORDS + base as usize / 2;
                    let end = begin + participants * width as usize;
                    assert_eq!(
                        bits(&words[begin..end]),
                        bits(&input[begin..end]),
                        "projection inputs modified"
                    );
                }
            }
            let words = &actual[0];
            let query_begin = GUARD_WORDS + layout.query as usize / 2;
            let query =
                &words[query_begin..query_begin + participants * shape.query_features as usize];
            let context_begin = GUARD_WORDS + layout.context as usize / 2;
            let context =
                &words[context_begin..context_begin + participants * shape.query_features as usize];
            assert!(query.iter().chain(context).all(|value| value.is_finite()));
            let query = bits(query);
            if let Some(previous) = previous_query.replace(query.clone()) {
                assert_ne!(previous, query, "replay ignored fresh projection data");
            }
            check_pages(&mut arms, &stream, shape, &requests, &lengths);
            query_norm.assert_unchanged(&stream);
            key_norm.assert_unchanged(&stream);
        }
        stream.synchronize().unwrap();
        // Graphs die before all pointed-to allocations. No driver-fault or
        // failed-drain lifetime guarantee is asserted by this success oracle.
        drop(graphs);
        println!("packed_prepare_gate_oracle indexed_grouping={indexed} head_dim={head_dim} participants={participants} output_gate={output_gate} physical_page_tokens={page_tokens} complete_query_context_kv_bits_equal=true current_binding_growth_and_new_request_allocations=true shared_allocation_disjoint_page_ranges=true");
    }
}

#[allow(clippy::too_many_arguments)]
fn capture_batches(
    stream: &Arc<CudaStream>,
    functions: &CausalAttentionFunctions,
    arms: &[Arm; 2],
    batches: &[BatchDecode; 2],
    launches: &[CausalAttentionLaunch],
    binding_layout: BindingLayout,
    cuda_shape: CudaCausalAttentionShape,
    layout: ScratchLayout,
    query_norm: &Guarded<f16>,
    key_norm: &Guarded<f16>,
    output_gate: bool,
) -> Vec<cudarc::driver::CudaGraph> {
    arms.iter()
        .zip(batches)
        .map(|(arm, batch)| {
            stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .unwrap();
            batch
                .enqueue(
                    &stream,
                    functions,
                    launches,
                    arm.binding_pointer(&stream),
                    binding_layout,
                    cuda_shape,
                    layout,
                    arm.scratch_pointer(&stream),
                    query_norm.pointer(&stream),
                    key_norm.pointer(&stream),
                    output_gate,
                )
                .unwrap();
            stream
                .end_capture(
                    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
                .unwrap()
                .expect("nonempty full batch decode graph")
        })
        .collect::<Vec<_>>()
}
