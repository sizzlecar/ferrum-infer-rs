//! Exact sampling semantics and paired timing of large-vocabulary reductions.
use super::*;
use crate::backend::metal::vnext_ops::MetalVNextComposition;
use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};
use half::f16;
use metal::objc::{msg_send, sel, sel_impl};
use metal::{CommandBufferRef, CommandQueueRef, MTLCommandBufferStatus};

const GUARD: usize = 64;

struct Guarded {
    parent: MetalBufferRegion,
    region: MetalBufferRegion,
    initial: Vec<u8>,
}

impl Guarded {
    fn new(runtime: &MetalDeviceRuntime, index: usize, bytes: Vec<u8>) -> Self {
        let mut initial = vec![0xa5; GUARD + bytes.len() + GUARD];
        initial[GUARD..GUARD + bytes.len()].copy_from_slice(&bytes);
        let parent = runtime
            .allocate_test_region(
                &BufferRequest::new(
                    ResourceId::new(format!("argmax/{index}")).unwrap(),
                    initial.len() as u64,
                    64,
                    BufferUsage::Transfer,
                    ElementType::U8,
                )
                .unwrap(),
            )
            .unwrap();
        // SAFETY: the test exclusively owns this shared allocation and writes
        // exactly its admitted byte length before any command is submitted.
        unsafe {
            std::ptr::copy_nonoverlapping(
                initial.as_ptr(),
                parent
                    .buffer()
                    .contents()
                    .cast::<u8>()
                    .add(parent.offset_bytes() as usize),
                initial.len(),
            );
        }
        let region = parent
            .test_subregion(GUARD as u64..(GUARD + bytes.len()) as u64)
            .unwrap();
        Self {
            parent,
            region,
            initial,
        }
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: only called after the associated command has completed.
        unsafe {
            std::slice::from_raw_parts(
                self.parent
                    .buffer()
                    .contents()
                    .cast::<u8>()
                    .add(self.parent.offset_bytes() as usize),
                self.initial.len(),
            )
        }
    }

    fn check(&self, mutable: bool) {
        let actual = self.bytes();
        assert_eq!(&actual[..GUARD], &self.initial[..GUARD]);
        assert_eq!(
            &actual[actual.len() - GUARD..],
            &self.initial[actual.len() - GUARD..]
        );
        if !mutable {
            assert_eq!(actual, self.initial);
        }
    }
}

struct Fixture {
    buffers: Vec<Guarded>,
    params: LastTokenMaskedArgmaxParams,
    dtype: ElementType,
    expected: u32,
}

impl Fixture {
    fn new(runtime: &MetalDeviceRuntime, vocab: usize, dtype: ElementType, case: u8) -> Self {
        let round = |x: f32| {
            if dtype == ElementType::F16 {
                f16::from_f32(x).to_f32()
            } else {
                x
            }
        };
        let mut logits: Vec<f32> = (0..vocab)
            .map(|i| round((i % 251) as f32 * 0.03125 - 4.0))
            .collect();
        logits[255] = 8.0;
        logits[256] = 8.0;
        logits[vocab - 1] = 8.0;
        logits[17] = 32.0;
        logits[3] = f32::NAN;
        logits[4] = f32::INFINITY;
        logits[5] = f32::NEG_INFINITY;
        let mut mask = vec![1_u8; vocab];
        mask[17] = 0;
        if case == 1 {
            mask.fill(0);
        }
        if case == 2 {
            logits.fill(f32::NAN);
        }
        if case == 4 {
            logits.fill(-2.0);
            logits[255] = -0.25;
            logits[256] = -0.25;
            logits[300] = -0.5;
        }
        if case == 7 {
            logits.fill(0.0);
            logits[255] = 1.0;
            // F16 repetition processing must round before the argmax tie.
            logits[300] = round(1.0 / 1.1);
        }
        let ids = [255_u32, 256, (vocab - 1) as u32, u32::MAX];
        let penalty = if case == 7 {
            1.1_f32
        } else if case >= 3 {
            4.0_f32
        } else {
            1.0
        };
        let offsets = match case {
            5 => [0_u32, 0],
            6 => [9, 20],
            _ => [0, ids.len() as u32],
        };
        let mut processed = logits.clone();
        let start = (offsets[0] as usize).min(ids.len());
        let end = (offsets[1] as usize).min(ids.len());
        if penalty != 1.0 {
            for &token in &ids[start..end] {
                if let Some(value) = processed.get_mut(token as usize) {
                    if value.is_finite() {
                        *value = round(if *value > 0.0 {
                            *value / penalty
                        } else {
                            *value * penalty
                        });
                    }
                }
            }
        }
        let mut expected = u32::MAX;
        let mut maximum = f32::NEG_INFINITY;
        for (token, (&value, &valid)) in processed.iter().zip(&mask).enumerate() {
            if valid != 0 && value.is_finite() && (expected == u32::MAX || value > maximum) {
                expected = token as u32;
                maximum = value;
            }
        }
        let encoded = if dtype == ElementType::F16 {
            logits
                .iter()
                .flat_map(|&x| f16::from_f32(x).to_bits().to_ne_bytes())
                .collect()
        } else {
            logits.iter().flat_map(|x| x.to_ne_bytes()).collect()
        };
        let values = [
            encoded,
            mask,
            ids.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            offsets.iter().flat_map(|x| x.to_ne_bytes()).collect(),
            penalty.to_ne_bytes().to_vec(),
            0xdeadbeef_u32.to_ne_bytes().to_vec(),
            vec![0x5a; masked_argmax_scratch_stride(vocab as u64, dtype).unwrap() as usize],
        ];
        Self {
            buffers: values
                .into_iter()
                .enumerate()
                .map(|(i, b)| Guarded::new(runtime, i, b))
                .collect(),
            params: LastTokenMaskedArgmaxParams {
                vocabulary_size: vocab as u32,
                repetition_capacity: ids.len() as u32,
            },
            dtype,
            expected,
        }
    }

    fn encode(
        &self,
        pipelines: &MetalPrimitivePipelines,
        encoder: &ComputeCommandEncoderRef,
        candidate: bool,
    ) {
        let r: Vec<_> = self.buffers.iter().map(|b| &b.region).collect();
        if candidate {
            dispatch_last_token_masked_argmax(
                pipelines,
                encoder,
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                r[6],
                0,
                self.params,
                self.dtype,
            );
        } else {
            encoder.set_compute_pipeline_state(if self.dtype == ElementType::F16 {
                &pipelines.last_token_masked_argmax
            } else {
                &pipelines.last_token_masked_argmax_f32
            });
            for (slot, region) in [r[0], r[6], r[1], r[2], r[3], r[4], r[5]]
                .into_iter()
                .enumerate()
            {
                set_region(encoder, slot as u64, region);
            }
            encoder.set_bytes(
                7,
                std::mem::size_of_val(&self.params) as u64,
                &self.params as *const _ as *const c_void,
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(1, 1, 1),
                MTLSize::new(THREADS_PER_GROUP, 1, 1),
            );
        }
    }

    fn run(
        &self,
        pipelines: &MetalPrimitivePipelines,
        queue: &CommandQueueRef,
        candidate: bool,
        iterations: usize,
    ) -> (f64, f64) {
        // Reset just the output. Keep prior scratch contents to exercise reuse
        // after both the original and partitioned algorithms.
        unsafe {
            std::ptr::write(
                self.buffers[5]
                    .region
                    .buffer()
                    .contents()
                    .cast::<u8>()
                    .add(self.buffers[5].region.offset_bytes() as usize)
                    .cast::<u32>(),
                0xdeadbeef,
            );
        }
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for _ in 0..iterations {
            self.encode(pipelines, encoder, candidate);
        }
        encoder.end_encoding();
        let start = std::time::Instant::now();
        command.commit();
        command.wait_until_completed();
        let wall = start.elapsed().as_secs_f64() * 1e9;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let gpu = gpu_ns(command);
        let bytes = self.buffers[5].bytes();
        assert_eq!(
            u32::from_ne_bytes(bytes[GUARD..GUARD + 4].try_into().unwrap()),
            self.expected
        );
        for (i, buffer) in self.buffers.iter().enumerate() {
            buffer.check(i >= 5);
        }
        (gpu / iterations as f64, wall / iterations as f64)
    }
}

#[allow(unexpected_cfgs, reason = "objc macro legacy feature checks")]
fn gpu_ns(command: &CommandBufferRef) -> f64 {
    // SAFETY: command completion was checked before reading documented Metal
    // command-buffer timestamps (double-valued properties).
    let (start, end): (f64, f64) = unsafe {
        (
            msg_send![command, GPUStartTime],
            msg_send![command, GPUEndTime],
        )
    };
    assert!(start.is_finite() && end.is_finite() && end > start && start > 0.0);
    (end - start) * 1e9
}

#[test]
fn parallel_argmax_preserves_masks_ties_nonfinite_penalties_and_regions_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.argmax.test").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalPrimitivePipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for dtype in [ElementType::F16, ElementType::F32] {
        for vocab in [8191, 8192, 8193, 248320] {
            for case in 0..8 {
                let fixture = Fixture::new(runtime, vocab, dtype, case);
                fixture.run(&pipelines, &queue, false, 1);
                fixture.run(&pipelines, &queue, true, 1);
                fixture.run(&pipelines, &queue, true, 1);
            }
        }
    }
}

#[test]
fn parallel_argmax_keeps_participant_scratch_windows_isolated_on_metal() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.argmax.slots").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalPrimitivePipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for dtype in [ElementType::F16, ElementType::F32] {
        let first = Fixture::new(runtime, 8193, dtype, 0);
        let second = Fixture::new(runtime, 8193, dtype, 7);
        let stride = masked_argmax_scratch_stride(8193, dtype).unwrap();
        let scratch = Guarded::new(runtime, 9, vec![0x5a; (stride * 2) as usize]);
        for order in [[0_usize, 1], [1, 0]] {
            let command = queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            for index in order {
                let fixture = [&first, &second][index];
                let r: Vec<_> = fixture.buffers.iter().map(|b| &b.region).collect();
                dispatch_last_token_masked_argmax(
                    &pipelines,
                    encoder,
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                    r[4],
                    r[5],
                    &scratch.region,
                    index as u64 * stride,
                    fixture.params,
                    dtype,
                );
            }
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            for fixture in [&first, &second] {
                let output = fixture.buffers[5].bytes();
                assert_eq!(
                    u32::from_ne_bytes(output[GUARD..GUARD + 4].try_into().unwrap()),
                    fixture.expected
                );
                for (i, buffer) in fixture.buffers.iter().enumerate() {
                    buffer.check(i == 5);
                }
            }
            scratch.check(true);
            let bytes = scratch.bytes();
            assert!(bytes[GUARD..GUARD + 256].iter().any(|&b| b != 0x5a));
            assert!(bytes[GUARD + 256..GUARD + stride as usize]
                .iter()
                .all(|&b| b == 0x5a));
        }
    }
}

#[test]
fn parallel_argmax_workspace_covers_partials_without_expanding_capacity() {
    for dtype in [ElementType::F16, ElementType::F32] {
        for vocab in [1, 8191, 8192, 8193, u32::MAX] {
            let dispatches = masked_argmax_dispatch_count(vocab);
            assert_eq!(
                dispatches,
                if vocab < MASKED_ARGMAX_PARALLEL_MIN_VOCAB {
                    1
                } else {
                    2
                }
            );
            if dispatches == 2 {
                assert!(
                    masked_argmax_scratch_stride(u64::from(vocab), dtype).unwrap()
                        >= MASKED_ARGMAX_PARTITIONS * 8
                );
            }
        }
    }
}

#[test]
#[ignore = "paired Metal GPU timing; requires exclusive device access"]
fn parallel_argmax_dispatch_microbench() {
    let composition =
        MetalVNextComposition::create(DeviceId::new("metal.argmax.bench").unwrap()).unwrap();
    let runtime = composition.runtime();
    let pipelines = MetalPrimitivePipelines::new(runtime.device()).unwrap();
    let queue = runtime.device().new_command_queue();
    for dtype in [ElementType::F16, ElementType::F32] {
        for vocab in [8192, 248320] {
            for case in [0, 3] {
                let fixture = Fixture::new(runtime, vocab, dtype, case);
                for round in 0..10 {
                    for candidate in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let (gpu, wall) = fixture.run(&pipelines, &queue, candidate, 32);
                        if round >= 2 {
                            println!(
                                "{}",
                                serde_json::json!({
                                    "benchmark":"parallel_masked_argmax","dtype":format!("{dtype:?}"),
                                    "vocab":vocab,"penalty_active":case==3,"candidate":candidate,
                                    "round":round-2,"iterations":32,"gpu_ns":gpu,"wall_ns":wall,
                                    "physical_dispatches":if candidate {masked_argmax_dispatch_count(vocab as u32)} else {1}
                                })
                            );
                        }
                    }
                }
            }
        }
    }
}
