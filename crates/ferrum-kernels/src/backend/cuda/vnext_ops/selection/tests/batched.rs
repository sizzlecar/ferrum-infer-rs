//! Real multi-row selection and complete sampler-to-host diagnostics. The
//! readback remains scalar: this probe does not implement a gather operation.
use super::*;
use cudarc::driver::{result, CudaSlice, DevicePtr};
use std::time::Instant;

#[path = "../../native_blocks/tests/captured.rs"]
mod captured;

struct Row {
    logits: CudaSlice<u8>,
    valid: CudaSlice<u8>,
    repetition: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    penalty: CudaSlice<f32>,
    output: CudaSlice<u32>,
    case: Case,
    penalty_value: f32,
}

struct Fixture {
    rows: Vec<Row>,
    scratch: CudaSlice<u8>,
    scratch_stride: usize,
    precision: ArgmaxPrecision,
    vocabulary: usize,
}

impl Fixture {
    fn new(
        stream: &Arc<cudarc::driver::CudaStream>,
        precision: ArgmaxPrecision,
        width: usize,
        vocabulary: usize,
    ) -> Self {
        let cases = partitioned::large_cases(vocabulary);
        let scratch_stride =
            masked_argmax_scratch_stride(vocabulary as u64, precision.element()).unwrap() as usize;
        let rows = (0..width)
            .map(|index| {
                let case = cases[index % cases.len()].clone();
                let penalty_value = [1.0_f32, 1.1, 0.75][(index + index / cases.len()) % 3];
                Row {
                    logits: stream
                        .clone_htod(&partitioned::bytes(&case, precision))
                        .unwrap(),
                    valid: stream.clone_htod(&case.valid).unwrap(),
                    repetition: stream.clone_htod(&case.repetition).unwrap(),
                    offsets: stream.clone_htod(&case.offsets).unwrap(),
                    penalty: stream.clone_htod(&[penalty_value]).unwrap(),
                    output: stream.clone_htod(&[0xDEADBEEF_u32; 3]).unwrap(),
                    case,
                    penalty_value,
                }
            })
            .collect();
        Self {
            rows,
            scratch: stream
                .clone_htod(&vec![0xCD_u8; (scratch_stride + 32) * width])
                .unwrap(),
            scratch_stride,
            precision,
            vocabulary,
        }
    }

    fn arguments(&self, stream: &cudarc::driver::CudaStream) -> Vec<ArgmaxArguments> {
        self.rows
            .iter()
            .enumerate()
            .map(|(index, row)| ArgmaxArguments {
                logits: row.logits.device_ptr(stream).0,
                scratch: self.scratch.device_ptr(stream).0
                    + (index * (self.scratch_stride + 32) + 16) as u64,
                valid_mask: row.valid.device_ptr(stream).0,
                repetition_offsets: row.offsets.device_ptr(stream).0,
                repetition_token_ids: row.repetition.device_ptr(stream).0,
                repetition_penalty: row.penalty.device_ptr(stream).0,
                output: row.output.device_ptr(stream).0 + 4,
                vocabulary_size: self.vocabulary as i32,
                repetition_capacity: row.case.repetition.len() as i32,
            })
            .collect()
    }

    fn enqueue(
        &self,
        stream: &cudarc::driver::CudaStream,
        functions: &ArgmaxFunctions,
        args: &[ArgmaxArguments],
        batched: bool,
    ) {
        if batched {
            let table = ArgmaxBatchArguments::new(args).unwrap();
            functions
                .launch_batched(stream, &table, self.rows.len() as u32)
                .unwrap();
        } else {
            for &row in args {
                functions.launch(stream, row, true).unwrap();
            }
        }
    }

    fn readback(&self, stream: &cudarc::driver::CudaStream) -> Vec<u32> {
        self.rows
            .iter()
            .map(|row| {
                // Match the retained runtime's per-output stream synchronization
                // and four-byte synchronous copy. No coalescing in either arm.
                stream.synchronize().unwrap();
                let mut selected = [0_u32];
                unsafe {
                    result::memcpy_dtoh_sync(&mut selected, row.output.device_ptr(stream).0 + 4)
                }
                .unwrap();
                selected[0]
            })
            .collect()
    }

    fn expected(&self) -> Vec<u32> {
        self.rows
            .iter()
            .map(|row| reference(&row.case, self.precision, row.penalty_value))
            .collect()
    }

    fn verify_storage(&self, stream: &cudarc::driver::CudaStream) {
        let scratch = stream.clone_dtoh(&self.scratch).unwrap();
        for (index, row) in self.rows.iter().enumerate() {
            assert_eq!(
                stream.clone_dtoh(&row.logits).unwrap(),
                partitioned::bytes(&row.case, self.precision),
                "semantic logits changed"
            );
            let output = stream.clone_dtoh(&row.output).unwrap();
            assert_eq!(output[0], 0xDEADBEEF);
            assert_eq!(output[2], 0xDEADBEEF);
            let base = index * (self.scratch_stride + 32);
            assert!(scratch[base..base + 16]
                .iter()
                .chain(&scratch[base + 16 + self.scratch_stride..base + 32 + self.scratch_stride])
                .all(|&byte| byte == 0xCD));
            let capacity = row.case.repetition.len() as u32;
            let active = row.penalty_value != 1.0
                && row.case.offsets[0].min(capacity) < row.case.offsets[1].min(capacity);
            let written = if active {
                self.vocabulary * self.precision.element().size_bytes() as usize
            } else {
                ARGMAX_PARTITIONS as usize * 8
            };
            assert!(
                scratch[base + 16 + written..base + 16 + self.scratch_stride]
                    .iter()
                    .all(|&byte| byte == 0xCD)
            );
        }
    }
}

#[test]
fn batched_argmax_parameter_abi_and_supported_domain() {
    assert_eq!(std::mem::size_of::<ArgmaxArguments>(), 64);
    assert_eq!(std::mem::align_of::<ArgmaxArguments>(), 8);
    assert_eq!(std::mem::offset_of!(ArgmaxArguments, vocabulary_size), 56);
    assert_eq!(
        std::mem::offset_of!(ArgmaxArguments, repetition_capacity),
        60
    );
    assert_eq!(std::mem::size_of::<ArgmaxBatchArguments>(), 2048);
    assert_eq!(std::mem::align_of::<ArgmaxBatchArguments>(), 8);
    let row = ArgmaxArguments {
        vocabulary_size: 248320,
        repetition_capacity: 4,
        ..Default::default()
    };
    for width in [2, 8, 32] {
        assert!(ArgmaxBatchArguments::new(&vec![row; width]).is_some());
    }
    for width in [0, 1, 33] {
        assert!(ArgmaxBatchArguments::new(&vec![row; width]).is_none());
    }
    assert!(ArgmaxBatchArguments::new(&[
        row,
        ArgmaxArguments {
            vocabulary_size: 65535,
            ..row
        }
    ])
    .is_none());
    assert!(ArgmaxBatchArguments::new(&[
        row,
        ArgmaxArguments {
            repetition_capacity: 0,
            ..row
        }
    ])
    .is_none());
}

#[test]
#[ignore = "requires an actual CUDA device"]
fn batched_argmax_matches_individual_rows_and_captured_graph_on_cuda() {
    let context = CudaContext::new(0).expect("batched selection requires CUDA");
    let stream = context.default_stream();
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let functions = ArgmaxFunctions::load(&context, precision).unwrap();
        for (width, vocabulary) in [(2, 65536), (32, 248320)] {
            let mut fixture = Fixture::new(&stream, precision, width, vocabulary);
            // Resolve test allocation handles before capture. Production also
            // launches from already-authorized raw CudaBufferRegion addresses.
            let arguments = fixture.arguments(&stream);
            let expected = fixture.expected();
            for batched in [false, true] {
                fixture.enqueue(&stream, &functions, &arguments, batched);
                assert_eq!(fixture.readback(&stream), expected);
                fixture.verify_storage(&stream);
                // The stack parameter table dies inside capture; the graph
                // must own its kernel arguments while fixture buffers survive.
                let graph = captured::Captured::new(&stream, || {
                    fixture.enqueue(&stream, &functions, &arguments, batched)
                });
                assert_eq!(graph.node_count, if batched { 2 } else { width * 2 });
                graph.launch();
                assert_eq!(fixture.readback(&stream), expected);
                fixture.verify_storage(&stream);
                if batched {
                    // Replay must read fresh per-row policies through the
                    // retained pointers, rather than capturing their values.
                    for row in &mut fixture.rows {
                        row.penalty_value = if row.penalty_value == 1.0 { 1.1 } else { 1.0 };
                        stream
                            .memcpy_htod(&[row.penalty_value], &mut row.penalty)
                            .unwrap();
                    }
                    stream
                        .memcpy_htod(
                            &vec![0xCD_u8; (fixture.scratch_stride + 32) * width],
                            &mut fixture.scratch,
                        )
                        .unwrap();
                    graph.launch();
                    assert_eq!(fixture.readback(&stream), fixture.expected());
                    fixture.verify_storage(&stream);
                }
                println!(
                    "{}",
                    serde_json::json!({"probe":"batched_argmax_correctness","precision":if matches!(precision,ArgmaxPrecision::F16){"f16"}else{"f32"},"width":width,"vocabulary":vocabulary,"batched":batched,"captured_nodes":graph.node_count,"explicit_parameter_htod":0,"parameter_table_bytes":if batched{2048}else{0},"scalar_token_dtoh_calls":width})
                );
            }
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device; emits complete terminal-transaction diagnostics"]
fn batched_argmax_m32_complete_terminal_transaction_diagnostic() {
    let context = CudaContext::new(0).expect("batched selection diagnostic requires CUDA");
    let stream = context.default_stream();
    for precision in [ArgmaxPrecision::F16, ArgmaxPrecision::F32] {
        let functions = ArgmaxFunctions::load(&context, precision).unwrap();
        let fixture = Fixture::new(&stream, precision, 32, 248320);
        let arguments = fixture.arguments(&stream);
        let expected = fixture.expected();
        let original = captured::Captured::new(&stream, || {
            fixture.enqueue(&stream, &functions, &arguments, false)
        });
        let candidate = captured::Captured::new(&stream, || {
            fixture.enqueue(&stream, &functions, &arguments, true)
        });
        assert_eq!((original.node_count, candidate.node_count), (64, 2));
        for round in 0..6 {
            for batched in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                for replay in [false, true] {
                    stream.synchronize().unwrap();
                    let wall = Instant::now();
                    if replay {
                        if batched {
                            candidate.launch();
                        } else {
                            original.launch();
                        }
                    } else {
                        // Includes fresh batch-table assembly, zero filling,
                        // parameter copying and every native kernel launch.
                        fixture.enqueue(&stream, &functions, &arguments, batched);
                    }
                    let selected = fixture.readback(&stream);
                    let wall_ns = wall.elapsed().as_nanos();
                    assert_eq!(selected, expected);
                    let graph = if batched { &candidate } else { &original };
                    let (_, gpu_ns) = captured::measure(&stream, || graph.launch());
                    if round >= 2 {
                        println!(
                            "{}",
                            serde_json::json!({"probe":"batched_argmax_terminal_transaction","precision":if matches!(precision,ArgmaxPrecision::F16){"f16"}else{"f32"},"width":32,"vocabulary":248320,"batched":batched,"graph_replay":replay,"round":round-2,"transaction_wall_ns":wall_ns,"separate_graph_gpu_ns":gpu_ns,"captured_nodes":graph.node_count,"explicit_parameter_htod":0,"scalar_token_dtoh_calls":32,"setup_excluded":"shared input/scratch allocation and graph capture/upload; no extra device allocation in candidate","wall_scope":"enqueue including direct parameter packing or graph replay, 32 per-output sync/4B DtoH, host result assembly; excludes Core authorization/model execution"})
                        );
                    }
                }
            }
        }
        fixture.verify_storage(&stream);
    }
}
