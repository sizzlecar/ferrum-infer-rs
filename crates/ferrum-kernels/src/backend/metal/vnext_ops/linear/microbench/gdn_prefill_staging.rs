//! Explicit fused versus fresh-stage QKVZBA algorithm comparison.
//! Production independently selects staging by input width and token count.
//! Q5 QKV and Q4 gate each pay fresh stage + GEMM. Both Q8 leaves stay unchanged.
use super::*;

mod validation;

const PREFIX: usize = 7;
const STAGED_PREFIX: usize = 8;
const SUFFIX: usize = 17;
const WEIGHT_PREFIX: usize = 16;
const GUARD: f16 = f16::from_bits(0x57b0);
const BYTE_GUARD: u8 = 0xa5;

#[derive(Clone, Copy)]
struct Geometry {
    rows: u32,
    input: u32,
    widths: [u32; 4],
}

impl Geometry {
    fn packed_width(self) -> u32 {
        self.widths.into_iter().sum()
    }

    fn shapes(self) -> [Shape; 4] {
        std::array::from_fn(|i| Shape {
            name: ["qkv", "gate", "beta", "alpha"][i],
            input: self.input,
            output: self.widths[i],
            format: [
                GgufBlockFormat::Q5K,
                GgufBlockFormat::Q4K,
                GgufBlockFormat::Q8_0,
                GgufBlockFormat::Q8_0,
            ][i],
        })
    }

    fn staging_elements(self) -> usize {
        self.input as usize * self.widths[0].max(self.widths[1]) as usize
    }
}

struct WeightSet {
    buffers: [Buffer; 4],
}

struct Case {
    geometry: Geometry,
    input: Buffer,
    input_values: Vec<f16>,
    encoded: [Vec<u8>; 4],
    weights: Vec<WeightSet>,
    outputs: Vec<[Buffer; 2]>,
    staged: Buffer,
}

impl Case {
    fn new(device: &Device, geometry: Geometry, worksets: usize) -> Self {
        assert!(worksets > 0);
        let mut input_values = vec![GUARD; PREFIX];
        input_values.extend(
            (0..geometry.rows as usize * geometry.input as usize)
                .map(|i| f16::from_f32((i as f32 * 0.013).sin() * 0.125)),
        );
        input_values.extend([GUARD; SUFFIX]);
        let encoded = geometry.shapes().map(weights);
        let weights = (0..worksets)
            .map(|_| WeightSet {
                buffers: std::array::from_fn(|i| {
                    let mut data = vec![BYTE_GUARD; WEIGHT_PREFIX];
                    data.extend_from_slice(&encoded[i]);
                    data.extend([BYTE_GUARD; SUFFIX]);
                    buffer(device, &data)
                }),
            })
            .collect();
        let output_elements =
            PREFIX + geometry.rows as usize * geometry.packed_width() as usize + SUFFIX;
        let outputs = (0..worksets)
            .map(|_| std::array::from_fn(|_| buffer(device, &vec![GUARD; output_elements])))
            .collect();
        let staged = buffer(
            device,
            &vec![GUARD; STAGED_PREFIX + geometry.staging_elements() + SUFFIX],
        );
        Self {
            geometry,
            input: buffer(device, &input_values),
            input_values,
            encoded,
            weights,
            outputs,
            staged,
        }
    }

    fn reset(&self) {
        // SAFETY: this fixture owns the shared buffers and waits for every
        // submitted command before touching their memory on the host.
        unsafe {
            for output in self.outputs.iter().flatten() {
                let values = std::slice::from_raw_parts_mut(
                    output.contents().cast::<f16>(),
                    output.length() as usize / 2,
                );
                values.fill(GUARD);
                let len = values.len();
                values[PREFIX..len - SUFFIX].fill(f16::NAN);
            }
            let stage = std::slice::from_raw_parts_mut(
                self.staged.contents().cast::<f16>(),
                self.staged.length() as usize / 2,
            );
            stage.fill(GUARD);
            stage[STAGED_PREFIX..STAGED_PREFIX + self.geometry.staging_elements()].fill(f16::NAN);
        }
    }

    fn encode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipelines: &MetalLinearPipelines,
        set: usize,
        candidate: bool,
    ) {
        let mut column = 0;
        for (i, shape) in self.geometry.shapes().into_iter().enumerate() {
            let format = match shape.format {
                GgufBlockFormat::Q8_0 => LinearPhysicalFormat::Q8_0,
                other => physical(other),
            };
            let params = LinearParams {
                rows: self.geometry.rows,
                in_features: shape.input,
                out_features: shape.output,
                output_stride: self.geometry.packed_width(),
                output_column_offset: column,
            };
            let stage = candidate && i < 2;
            let (pipeline, dispatch) = if stage {
                let blocks =
                    u32::try_from(u64::from(shape.input) * u64::from(shape.output) / 256).unwrap();
                encoder.set_compute_pipeline_state(if i == 0 {
                    &pipelines.k_quant_gemm.stage_q5_k
                } else {
                    &pipelines.k_quant_gemm.stage_q4_k
                });
                encoder.set_threadgroup_memory_length(0, 0);
                encoder.set_buffer(0, Some(&self.weights[set].buffers[i]), WEIGHT_PREFIX as u64);
                encoder.set_buffer(1, Some(&self.staged), (STAGED_PREFIX * 2) as u64);
                encoder.set_bytes(2, 4, (&blocks as *const u32).cast());
                encoder.dispatch_thread_groups(
                    MTLSize::new(u64::from(blocks).div_ceil(8), 1, 1),
                    MTLSize::new(128, 1, 1),
                );
                (
                    &pipelines.k_quant_gemm.staged_f16,
                    LinearDispatchKind::TiledGemm,
                )
            } else {
                pipelines.plain_linear_dispatch(format, ElementType::F16, params)
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&self.input), (PREFIX * 2) as u64);
            encoder.set_buffer(
                1,
                Some(if stage {
                    &self.staged
                } else {
                    &self.weights[set].buffers[i]
                }),
                if stage {
                    (STAGED_PREFIX * 2) as u64
                } else {
                    WEIGHT_PREFIX as u64
                },
            );
            encoder.set_buffer(
                2,
                Some(&self.outputs[set][usize::from(candidate)]),
                (PREFIX * 2) as u64,
            );
            bind_linear_params(encoder, params, format, ElementType::F16);
            dispatch_linear_grid(encoder, params, dispatch);
            column += shape.output;
        }
    }

    fn run(
        &self,
        queue: &CommandQueueRef,
        pipelines: &MetalLinearPipelines,
        candidate: bool,
    ) -> serde_json::Value {
        let started = Instant::now();
        let command = queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for set in 0..self.weights.len() {
            self.encode(encoder, pipelines, set, candidate);
        }
        encoder.end_encoding();
        let host_encode_ns = started.elapsed().as_nanos() as u64;
        let submitted = Instant::now();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        serde_json::json!({
            "candidate":candidate,
            "variant":if candidate {"fresh_q5_q4_stage_plus_f16_gemm_q8_unchanged"} else {"fused_four_leaves_reference"},
            "physical_dispatches": self.weights.len() * if candidate {6} else {4},
            "complete_qkvzba_projections":self.weights.len(),
            "host_encode_ns":host_encode_ns,
            "host_submit_wait_ns":submitted.elapsed().as_nanos() as u64,
            "device_command_ns":gpu_elapsed_ns(command),
        })
    }
}

#[test]
fn gdn_prefill_staging_dense_partition_tail_oracle() {
    let device = Device::system_default().expect("GDN staging conformance requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    // Dense full-output CPU oracle; M and N tiles have exact and partial tails.
    for rows in [31, 32, 33] {
        let case = Case::new(
            &device,
            Geometry {
                rows,
                input: 256,
                widths: [129, 65, 32, 32],
            },
            1,
        );
        case.reset();
        for candidate in [false, true] {
            case.run(&queue, &pipelines, candidate);
        }
        case.validate(true);
    }
}

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn gdn_prefill_staging_618_complete_projection_microbench() {
    measure_complete_projection(618);
}

#[test]
#[ignore = "GPU performance experiment: coordinate exclusive device access"]
fn gdn_prefill_staging_enable_boundaries_microbench() {
    for rows in [512, 767] {
        measure_complete_projection(rows);
    }
}

fn measure_complete_projection(rows: u32) {
    // Force both algorithms explicitly. The reference is not necessarily
    // the production route at this geometry after a staging-policy change.
    let device = Device::system_default().expect("GDN staging microbench requires Metal");
    let queue = device.new_command_queue();
    let pipelines = MetalLinearPipelines::new(&device).unwrap();
    let geometry = Geometry {
        rows,
        input: 4096,
        widths: [8192, 4096, 32, 32],
    };
    for worksets in [1, 4] {
        let case = Case::new(&device, geometry, worksets);
        case.reset();
        for candidate in [false, true] {
            case.run(&queue, &pipelines, candidate);
        }
        // No timing loop unless every packed output is bitwise equivalent,
        // independent CPU checkpoints agree, and all guards/inputs survive.
        let before = case.validate(false);
        let mut samples = Vec::new();
        for round in 0..8 {
            for (order, candidate) in if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            }
            .into_iter()
            .enumerate()
            {
                let mut sample = case.run(&queue, &pipelines, candidate);
                sample["round"] = round.into();
                sample["order"] = order.into();
                sample["warmup"] = (round < 2).into();
                samples.push(sample);
            }
        }
        let after = case.validate(false);
        println!(
            "{}",
            serde_json::json!({
                "kind":"gdn_prefill_staging_complete_projection",
                "device":device.name(),"rows":geometry.rows,"input_features":geometry.input,
                "leaf_widths":geometry.widths,"leaf_formats":["Q5_K","Q4_K","Q8_0","Q8_0"],
                "packed_output_stride":geometry.packed_width(),"worksets":worksets,
                "weight_contents":"identical_synthetic_coefficients_independent_allocations",
                "weight_payload_bytes":case.encoded.iter().map(Vec::len).sum::<usize>() * worksets,
                "weight_sha256":case.encoded.iter().map(|bytes|format!("{:x}",Sha256::digest(bytes))).collect::<Vec<_>>(),
                "fresh_staging_payload_bytes_per_projection":(u64::from(geometry.widths[0])+u64::from(geometry.widths[1]))*u64::from(geometry.input)*2,
                "shared_staging_allocation_bytes":case.staged.length(),
                "fresh_stage_in_every_candidate":true,
                "scope":"complete_four_leaf_input_projection_not_recurrence_output_projection_or_model_throughput",
                "numerical_before":before,"numerical_after":after,
                "engineering_screen_max_gpu_ratio":0.95,
                "paired_gpu_summary":paired_summary(&samples),
                "samples":samples,
            })
        );
    }
}

fn paired_summary(samples: &[serde_json::Value]) -> serde_json::Value {
    let mut control = Vec::new();
    let mut candidate = Vec::new();
    let mut pairs = Vec::new();
    let mut unavailable_pairs = 0;
    for pair in samples.chunks_exact(2) {
        if pair[0]["warmup"].as_bool() == Some(true) {
            continue;
        }
        let (a, b) = if pair[0]["candidate"].as_bool() == Some(false) {
            (&pair[0], &pair[1])
        } else {
            (&pair[1], &pair[0])
        };
        let times = a["device_command_ns"]
            .as_f64()
            .zip(b["device_command_ns"].as_f64())
            .filter(|(a, b)| a.is_finite() && b.is_finite() && *a > 0.0 && *b > 0.0);
        if let Some((a, b)) = times {
            control.push(a);
            candidate.push(b);
            pairs.push(serde_json::json!({"round":pair[0]["round"],"candidate_over_control":b/a}));
        } else {
            unavailable_pairs += 1;
            pairs.push(serde_json::json!({"round":pair[0]["round"],"candidate_over_control":null}));
        }
    }
    let means = (!control.is_empty() && unavailable_pairs == 0).then(|| {
        (
            control.iter().sum::<f64>() / control.len() as f64,
            candidate.iter().sum::<f64>() / candidate.len() as f64,
        )
    });
    serde_json::json!({
        "aggregation":"ratio_of_arithmetic_mean_complete_command_gpu_intervals",
        "control_mean_ns":means.map(|(a,_)|a),
        "candidate_mean_ns":means.map(|(_,b)|b),
        "candidate_over_control":means.map(|(a,b)|b/a),
        "engineering_screen_met":means.map(|(a,b)|b/a<=0.95),
        "unavailable_pairs":unavailable_pairs,"paired_ratios":pairs,
        "statistical_inference":"not_performed_not_a_service_or_SLO_acceptance",
    })
}
