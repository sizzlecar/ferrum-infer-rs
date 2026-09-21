//! Same prepared Q/K/V, original per-row wrappers versus batched wrappers.
use super::*;
use metal::objc::rc::autoreleasepool;
use std::time::Instant;

struct Fixture {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: MetalCausalAttentionPipelines,
    scratch: Buffer,
    initial: Vec<u8>,
    bindings: Buffer,
    rows: Vec<GroupedDecodeRow>,
    pages: Vec<Buffer>,
    page_bytes: Vec<Vec<u8>>,
    expected: Vec<Vec<f32>>,
}

fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // All callers use u8/f16/f32 with no uninitialized padding.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn segment(storage: &mut Vec<u8>, data: &[u8]) -> u64 {
    storage.resize(storage.len().next_multiple_of(16) + 64, 0x55);
    let offset = storage.len() as u64;
    storage.extend_from_slice(data);
    offset
}

impl Fixture {
    fn new(contexts: &[usize], dim: usize, heads: usize, kv_heads: usize) -> Self {
        let device = Device::system_default().expect("batched attention requires Metal");
        let queue = device.new_command_queue();
        let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
        let slot = pipelines.binding_slot_bytes().unwrap();
        let bindings = device.new_buffer(
            slot * (contexts.len() as u64 + 1),
            MTLResourceOptions::StorageModeShared,
        );
        let mut initial = vec![0x55; 256];
        let mut rows = Vec::new();
        let mut pages = Vec::new();
        let mut page_bytes = Vec::new();
        let mut expected = Vec::new();
        for (row, &context) in contexts.iter().enumerate() {
            let query = (0..heads * dim)
                .map(|i| f16::from_f32(((i + row * 7) as f32 * 0.013).sin() * 0.2))
                .collect::<Vec<_>>();
            let raw = (0..heads * dim * 2)
                .map(|i| f16::from_f32(((i + row * 13) as f32 * 0.017).cos() * 0.3))
                .collect::<Vec<_>>();
            let page_elements = VNEXT_KV_PAGE_BYTES as usize / 2;
            let valid = context * 2 * kv_heads * dim;
            let mut state = vec![f16::NAN; valid.next_multiple_of(page_elements)];
            for (i, value) in state[..valid].iter_mut().enumerate() {
                *value = f16::from_f32(((i + row * 29) as f32 * 0.0091).sin() * 0.25);
            }
            let first_page = pages.len();
            for page in state.chunks_exact(page_elements) {
                let mut guarded = vec![0x55_u8; 16];
                guarded.extend_from_slice(bytes(page));
                guarded.extend_from_slice(&[0x55; 16]);
                pages.push(shared_buffer(&device, &guarded));
                page_bytes.push(guarded);
            }
            let binding = (row as u64 + 1) * slot;
            pipelines
                .with_binding_encoder(|encoder| {
                    encoder.set_argument_buffer(&bindings, binding);
                    encoder.set_buffers(
                        0,
                        &pages[first_page..].iter().map(|p| &**p).collect::<Vec<_>>(),
                        &vec![16; pages.len() - first_page],
                    );
                    Ok(())
                })
                .unwrap();
            let params = CausalAttentionParams {
                page_elements: page_elements as u32,
                page_count: (pages.len() - first_page) as u32,
                position_start: context as u32 - 1,
                tokens: 1,
                query_heads: heads as u32,
                key_value_heads: kv_heads as u32,
                head_dim: dim as u32,
                rope_dim: dim as u32,
                query_projection_stride: (heads * dim * 2) as u32,
                query_head_stride: (dim * 2) as u32,
                kv_projection_stride: (kv_heads * dim) as u32,
                output_gate: 1,
                rope_interleaved: 0,
                attention_simdgroups: 4,
                epsilon: 1e-6,
                rope_theta: 10000.0,
            };
            assert_eq!(
                pipelines.dispatch_plan(&params).kind,
                AttentionDispatchKind::GroupedDecode
            );
            assert!(pipelines
                .specialization(&params)
                .unwrap()
                .batched_grouped
                .is_some());
            rows.push(GroupedDecodeRow {
                params,
                query: segment(&mut initial, bytes(&query)),
                query_raw: segment(&mut initial, bytes(&raw)),
                partials: segment(&mut initial, bytes(&vec![f32::NAN; 32 * heads * (dim + 2)])),
                output: segment(&mut initial, bytes(&vec![f16::NAN; heads * dim])),
                binding,
            });
            expected.push(cpu_tiled_prefill_attention(
                &query,
                &raw,
                &state,
                context - 1,
                1,
                heads,
                kv_heads,
                dim,
                true,
            ));
        }
        initial.extend_from_slice(&[0x55; 128]);
        let scratch = shared_buffer(&device, &initial);
        Self {
            device,
            queue,
            pipelines,
            scratch,
            initial,
            bindings,
            rows,
            pages,
            page_bytes,
            expected,
        }
    }

    fn run(&self, batched: bool) -> (Vec<Vec<u16>>, serde_json::Value) {
        // Every prior submit has completed. Poison all writable data before
        // timing; no old result can hide a missing candidate write.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.initial.as_ptr(),
                self.scratch.contents().cast(),
                self.initial.len(),
            );
        }
        let started = Instant::now();
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        use_raw_pages(encoder, &self.pages);
        if batched {
            for rows in self.rows.chunks(GROUPED_BATCH_ROWS) {
                // Both base views are nonzero as well as every inner offset.
                let adjusted = rows
                    .iter()
                    .map(|row| GroupedDecodeRow {
                        query: row.query - 128,
                        query_raw: row.query_raw - 128,
                        partials: row.partials - 128,
                        output: row.output - 128,
                        binding: row.binding - self.pipelines.binding_slot_bytes().unwrap(),
                        ..*row
                    })
                    .collect::<Vec<_>>();
                dispatch_batched_grouped_decode(
                    &self.pipelines,
                    encoder,
                    &self.scratch,
                    128,
                    &self.bindings,
                    self.pipelines.binding_slot_bytes().unwrap(),
                    &adjusted,
                );
            }
        } else {
            for row in &self.rows {
                encoder.set_buffer(0, Some(&self.scratch), row.query);
                encoder.set_buffer(1, Some(&self.scratch), row.query_raw);
                encoder.set_buffer(2, Some(&self.scratch), row.partials);
                encoder.set_buffer(
                    ATTENTION_PAGE_TABLE_INDEX,
                    Some(&self.bindings),
                    row.binding,
                );
                set_raw_params(encoder, 4, &row.params);
                encode_attention_dispatch(
                    &self.pipelines,
                    encoder,
                    grouped_decode_attention_dispatch_plan(&row.params),
                    &row.params,
                );
                encoder.set_compute_pipeline_state(
                    self.pipelines.grouped_reduce_pipeline(&row.params),
                );
                encoder.set_buffer(0, Some(&self.scratch), row.partials);
                encoder.set_buffer(1, Some(&self.scratch), row.query_raw);
                encoder.set_buffer(2, Some(&self.scratch), row.output);
                encoder.set_threadgroup_memory_length(
                    0,
                    grouped_decode_reduce_threadgroup_memory_bytes(),
                );
                encoder.dispatch_thread_groups(
                    MTLSize::new(row.params.query_heads as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
                encoder.set_threadgroup_memory_length(0, 0);
            }
        }
        encoder.end_encoding();
        command.commit();
        let encode_submit_ns = started.elapsed().as_nanos();
        command.wait_until_completed();
        let wall_ns = started.elapsed().as_nanos();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let mut timing = super::timing::gpu_clock(command);
        timing["encode_submit_ns"] = serde_json::json!(encode_submit_ns);
        timing["wall_ns"] = serde_json::json!(wall_ns);
        timing["batched"] = serde_json::json!(batched);
        let actual = unsafe {
            std::slice::from_raw_parts(self.scratch.contents().cast::<u8>(), self.initial.len())
        };
        let mut writable = vec![false; actual.len()];
        let mut result = Vec::new();
        for (row, expected) in self.rows.iter().zip(&self.expected) {
            let count = row.params.query_heads as usize * row.params.head_dim as usize;
            let partial_bytes = grouped_decode_partitions(&row.params) as usize
                * row.params.query_heads as usize
                * (row.params.head_dim as usize + 2)
                * 4;
            writable[row.partials as usize..row.partials as usize + partial_bytes].fill(true);
            writable[row.output as usize..row.output as usize + count * 2].fill(true);
            let values = unsafe {
                std::slice::from_raw_parts(
                    actual.as_ptr().add(row.output as usize).cast::<f16>(),
                    count,
                )
            };
            assert_close(
                "batched grouped independent CPU",
                &values.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
                expected,
                0.001,
            );
            result.push(values.iter().map(|x| x.to_bits()).collect());
        }
        for (i, (&actual, &initial)) in actual.iter().zip(&self.initial).enumerate() {
            if !writable[i] {
                assert_eq!(actual, initial, "scratch/input/unused-partial guard {i}");
            }
        }
        for (page, expected) in self.pages.iter().zip(&self.page_bytes) {
            let actual =
                unsafe { std::slice::from_raw_parts(page.contents().cast::<u8>(), expected.len()) };
            assert_eq!(
                actual, expected,
                "KV including nonzero offset and NaN slack must be unchanged"
            );
        }
        (result, timing)
    }
}

#[test]
fn batched_grouped_decode_matches_serial_bits_cpu_and_guards() {
    autoreleasepool(|| {
        for (contexts, dim, heads, kv) in [
            (vec![257, 529], 256, 16, 4),
            (vec![513, 1025, 257], 128, 8, 1),
            (vec![257; GROUPED_BATCH_ROWS + 1], 128, 4, 1),
        ] {
            let fixture = Fixture::new(&contexts, dim, heads, kv);
            let old = fixture.run(false).0;
            let new = fixture.run(true).0;
            assert_eq!(
                old, new,
                "all outputs must be bitwise equal across wrappers/chunks"
            );
        }
    });
}

#[test]
#[ignore = "exclusive Metal GPU; one N32 Q16/KV4/D256/context528 core diagnostic"]
fn batched_grouped_decode_microbench() {
    autoreleasepool(|| {
        let fixture = Fixture::new(&[528; 32], 256, 16, 4);
        let gate = GateFixture::new(&fixture.rows.iter().map(|r| r.params).collect::<Vec<_>>());
        eprintln!("device={}", fixture.device.name());
        let expected = fixture.run(false).0;
        assert_eq!(fixture.run(true).0, expected);
        for round in 0..3 {
            for batched in [false, true, true, false] {
                let gate_start = Instant::now();
                assert!(gate.eligible(&fixture.pipelines));
                let gate_ns = gate_start.elapsed().as_nanos();
                let (actual, mut timing) = fixture.run(batched);
                assert_eq!(actual, expected);
                timing["round"] = serde_json::json!(round);
                timing["production_selector_ns"] = serde_json::json!(gate_ns);
                eprintln!("BATCHED_GROUPED_MICRO {timing}");
            }
        }
    });
}

struct GateFixture {
    params: Vec<CausalAttentionParams>,
    pages: Vec<Vec<MetalBufferRegion>>,
}

impl GateFixture {
    fn new(params: &[CausalAttentionParams]) -> Self {
        use crate::backend::metal::vnext_runtime::MetalDeviceRuntimeConfig;
        use ferrum_interfaces::vnext::{BufferRequest, BufferUsage, DeviceId, ResourceId};
        let runtime = MetalDeviceRuntime::new(MetalDeviceRuntimeConfig {
            device_id: DeviceId::new("device/metal/batched-gate").unwrap(),
            runtime_implementation_fingerprint: "a".repeat(64),
            capabilities: Default::default(),
            dynamic_storage_profiles: std::collections::BTreeSet::from([
                DynamicStorageProfile::new(
                    DynamicStorageAllocator::LinearArena,
                    DynamicStorageView::Contiguous,
                )
                .unwrap(),
            ]),
        })
        .unwrap();
        let page_count = params.iter().map(|p| p.page_count as u64).sum::<u64>();
        let whole = runtime
            .allocate_test_region(
                &BufferRequest::new(
                    ResourceId::new("batched-gate.pages").unwrap(),
                    page_count * VNEXT_KV_PAGE_BYTES,
                    64,
                    BufferUsage::Transfer,
                    ElementType::U8,
                )
                .unwrap(),
            )
            .unwrap();
        let mut offset = 0;
        let pages = params
            .iter()
            .map(|p| {
                (0..p.page_count)
                    .map(|_| {
                        let start = offset;
                        offset += VNEXT_KV_PAGE_BYTES;
                        whole.test_subregion(start..offset).unwrap()
                    })
                    .collect()
            })
            .collect();
        Self {
            params: params.to_vec(),
            pages,
        }
    }

    fn eligible(&self, pipelines: &MetalCausalAttentionPipelines) -> bool {
        can_batch_grouped_decode(
            pipelines,
            self.params
                .iter()
                .zip(&self.pages)
                .map(|(p, pages)| (p, pages.as_slice(), 0)),
        )
    }
}

#[test]
fn batched_grouped_decode_production_gate_accepts_n32_and_rejects_physical_aliases() {
    let device = Device::system_default().unwrap();
    let pipelines = MetalCausalAttentionPipelines::new(&device).unwrap();
    let mut params = dispatch_test_params(1, 256);
    params.position_start = 527;
    params.page_count = ((528 * 2 * 4 * 256 * 2) as u64).div_ceil(VNEXT_KV_PAGE_BYTES) as u32;
    let mut gate = GateFixture::new(&[params; 32]);
    assert!(
        gate.eligible(&pipelines),
        "actual N32/528 page geometry must reach production route"
    );
    let saved = gate.pages[1][0].clone();
    gate.pages[1][0] = gate.pages[0][0].clone();
    assert!(
        !gate.eligible(&pipelines),
        "shared read-only prefix is conservatively serial"
    );
    gate.pages[1][0] = gate.pages[0][0].test_subregion(16..48).unwrap();
    assert!(
        !gate.eligible(&pipelines),
        "nested physical alias must not escape sorting"
    );
    gate.pages[1][0] = saved;
    assert!(
        gate.eligible(&pipelines),
        "adjacent nonoverlapping subviews remain eligible"
    );
    gate.params[3].tokens = 2;
    assert!(
        !gate.eligible(&pipelines),
        "mixed/prefill retains the original per-row path"
    );
    gate.params[3].tokens = 1;
    gate.params[4].head_dim = 128;
    assert!(!gate.eligible(&pipelines));
}
