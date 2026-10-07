//! Guarded pack/launch/timing fixture adapted from Ferrum 498e404f q4k_q8.rs.
//! The original Q4 pack ABI is retained; IQ4 and dot-only timing are additions.

use super::*;
use cudarc::driver::{sys::CUevent_flags, CudaSlice};
use q4k_reference::{fixture_block, pack_rows};

const INPUT_PREFIX: usize = 3;
const WEIGHT_PREFIX: usize = 5;
const SCALE_PREFIX: usize = 2;
const WORD_PREFIX: usize = 3;
const OUTPUT_PREFIX: usize = 4;
const COLUMN_OFFSET: usize = 2;

// Only the physical qword permutation changes; scales remain row/group major.
#[derive(Clone, Copy, Debug)]
enum QwordLayout {
    GroupMajor,
    WordMajor,
}

impl QwordLayout {
    fn physical(self, row: usize, group: usize, word: usize, groups: usize) -> usize {
        match self {
            Self::GroupMajor => (row * groups + group) * 8 + word,
            Self::WordMajor => (row * 8 + word) * groups + group,
        }
    }

    fn logical(self, physical: usize, groups: usize) -> usize {
        match self {
            Self::GroupMajor => physical,
            Self::WordMajor => {
                let row = physical / (8 * groups);
                let within_row = physical % (8 * groups);
                (row * groups + within_row % groups) * 8 + within_row / groups
            }
        }
    }
}

mod coalesced;
mod graph_paired;
mod q5k;

#[derive(Clone, Copy, Debug)]
pub(super) enum Route {
    Baseline,
    QuantizeAndDot,
    DotOnly,
    Group32QuantizeAndDot,
    Group32DotOnly,
}

impl Route {
    const Q5: [Self; 3] = [
        Self::Baseline,
        Self::Group32QuantizeAndDot,
        Self::Group32DotOnly,
    ];
    pub(super) const ALL: [Self; 5] = [
        Self::Baseline,
        Self::QuantizeAndDot,
        Self::DotOnly,
        Self::Group32QuantizeAndDot,
        Self::Group32DotOnly,
    ];
    const QUANTIZED_GROUPS: [[Self; 2]; 2] = [
        [Self::QuantizeAndDot, Self::DotOnly],
        [Self::Group32QuantizeAndDot, Self::Group32DotOnly],
    ];

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Baseline => "retained_f32",
            Self::QuantizeAndDot => "q8_pack_plus_dp4a",
            Self::DotOnly => "q8_dp4a_only",
            Self::Group32QuantizeAndDot => "q8_group32_pack_plus_dp4a",
            Self::Group32DotOnly => "q8_group32_dp4a_only",
        }
    }

    pub(super) fn includes_quantization(self) -> bool {
        matches!(self, Self::QuantizeAndDot | Self::Group32QuantizeAndDot)
    }

    pub(super) fn integer_partial_values(self) -> Option<u32> {
        match self {
            Self::Baseline => None,
            Self::QuantizeAndDot | Self::DotOnly => Some(4),
            Self::Group32QuantizeAndDot | Self::Group32DotOnly => Some(32),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Reference {
    original: f64,
    quantized: f64,
    policy: f64,
    original_abs: f64,
    expanded_abs: f64,
    activation_bound: f64,
}

pub(super) struct Fixture {
    format: GgufBlockFormat,
    rows: usize,
    inputs: usize,
    outputs: usize,
    stride: usize,
    input: Vec<f16>,
    weights: Vec<u8>,
    input_gpu: CudaSlice<f16>,
    weights_gpu: CudaSlice<u8>,
    scales_gpu: CudaSlice<f32>,
    words_gpu: CudaSlice<u32>,
    output_gpu: CudaSlice<f16>,
    references: Vec<Option<Reference>>,
}

impl Fixture {
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        format: GgufBlockFormat,
        rows: usize,
        inputs: usize,
        outputs: usize,
    ) -> Self {
        assert!(rows > 0 && inputs > 0 && inputs.is_multiple_of(32));
        assert!(matches!(
            format,
            GgufBlockFormat::Q4K | GgufBlockFormat::Q5K | GgufBlockFormat::Iq4Xs
        ));
        let mut input = vec![f16::from_f32(-12345.0); INPUT_PREFIX + rows * inputs + 5];
        for row in 0..rows {
            for col in 0..inputs {
                let x = if (col / 32 + row) % 7 == 0 {
                    0.0
                } else {
                    ((col * 13 + row * 17) % 67) as f32 / 127.0 - 0.25
                };
                input[INPUT_PREFIX + row * inputs + col] = f16::from_f32(x);
            }
        }
        let iq_seed = oracle_blocks(GgufBlockFormat::Iq4Xs);
        let mut weights = vec![0xcc; WEIGHT_PREFIX];
        for col in 0..outputs {
            for block in 0..inputs / 256 {
                match format {
                    GgufBlockFormat::Q4K => weights.extend(fixture_block(col, block).encode()),
                    GgufBlockFormat::Q5K => {
                        weights.extend(q5k_reference::fixture_q5(col, block).encode())
                    }
                    GgufBlockFormat::Iq4Xs => {
                        let seed = ((col + block) % 2) * 136;
                        let mut encoded = iq_seed[seed..seed + 136].to_vec();
                        // Reuse the shared GGUF fixture's codes/scales; vary the
                        // outer signed scale by column/block without huge outputs.
                        let magnitude = (1 + (col * 7 + block * 3) % 29) as f32 / 4096.0;
                        let scale = if (col + block) % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        };
                        encoded[..2].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                        weights.extend(encoded);
                    }
                    _ => unreachable!(),
                }
            }
        }
        weights.extend([0xcc; 7]);
        let groups = rows * inputs / 32;
        let stride = outputs + 5;
        let mut result = Self {
            format,
            rows,
            inputs,
            outputs,
            stride,
            input_gpu: stream.clone_htod(&input).unwrap(),
            weights_gpu: stream.clone_htod(&weights).unwrap(),
            scales_gpu: stream
                .clone_htod(&vec![-8765.25_f32; SCALE_PREFIX + groups + 5])
                .unwrap(),
            words_gpu: stream
                .clone_htod(&vec![0xaabbccdd_u32; WORD_PREFIX + groups * 8 + 5])
                .unwrap(),
            output_gpu: stream
                .clone_htod(&vec![
                    f16::from_f32(-12345.0);
                    OUTPUT_PREFIX + rows * stride + 5
                ])
                .unwrap(),
            input,
            weights,
            references: Vec::new(),
        };
        if inputs.is_multiple_of(256) {
            result.prepare_references();
        }
        result
    }

    // Independent CPU references are prepared outside every timed submission.
    // Full small correctness matrices are checked; large timing matrices use
    // the old fixture's first/middle/last row/column oracle sample. Every output
    // is still checked for a finite write, guards, and repeated-route bit stability.
    fn prepare_references(&mut self) {
        self.references = vec![None; self.rows * self.outputs];
        let packed = pack_rows(
            &self.input[INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs],
            self.rows,
            self.inputs,
        );
        for row in 0..self.rows {
            for col in 0..self.outputs {
                if self.outputs > 49
                    && (![0, self.rows / 2, self.rows - 1].contains(&row)
                        || ![0, self.outputs / 2, self.outputs - 1].contains(&col))
                {
                    continue;
                }
                let bytes_per_row = self.inputs / 256 * self.format.block_bytes();
                let bytes = &self.weights[WEIGHT_PREFIX + col * bytes_per_row..][..bytes_per_row];
                let mut decoded = vec![0.0_f32; self.inputs];
                self.format.decode(bytes, &mut decoded).unwrap();
                let mut reference = Reference::default();
                for (i, &weight) in decoded.iter().enumerate() {
                    let x = f64::from(self.input[INPUT_PREFIX + row * self.inputs + i].to_f32());
                    let xhat = f64::from(packed.scales[(row * self.inputs + i) / 32])
                        * f64::from(packed.quants[row * self.inputs + i]);
                    let weight = f64::from(weight);
                    reference.original += weight * x;
                    reference.original_abs += (weight * x).abs();
                    reference.quantized += weight * xhat;
                    reference.activation_bound += weight.abs() * (xhat - x).abs();
                }
                for group in 0..self.inputs / 32 {
                    let block = &bytes[(group / 8) * self.format.block_bytes()..]
                        [..self.format.block_bytes()];
                    let start = row * self.inputs + group * 32;
                    let (value, magnitude) = oracle::block_formula(
                        self.format,
                        block,
                        group % 8,
                        &packed.quants[start..start + 32],
                        packed.scales[start / 32],
                    );
                    reference.policy += value;
                    reference.expanded_abs += magnitude;
                }
                assert!(reference.original.is_finite() && reference.policy.is_finite());
                assert!(reference.original.abs() < f64::from(f16::MAX.to_f32()));
                assert!(reference.policy.abs() < f64::from(f16::MAX.to_f32()));
                self.references[row * self.outputs + col] = Some(reference);
            }
        }
    }

    fn reset(&mut self, stream: &Arc<CudaStream>) {
        let groups = self.rows * self.inputs / 32;
        let mut output = vec![f16::from_f32(-12345.0); OUTPUT_PREFIX + self.rows * self.stride + 5];
        for row in 0..self.rows {
            let start = OUTPUT_PREFIX + row * self.stride + COLUMN_OFFSET;
            output[start..start + self.outputs].fill(f16::NAN);
        }
        stream.memcpy_htod(&output, &mut self.output_gpu).unwrap();
        stream
            .memcpy_htod(
                &vec![-8765.25_f32; SCALE_PREFIX + groups + 5],
                &mut self.scales_gpu,
            )
            .unwrap();
        stream
            .memcpy_htod(
                &vec![0xaabbccdd_u32; WORD_PREFIX + groups * 8 + 5],
                &mut self.words_gpu,
            )
            .unwrap();
    }

    fn pack(&mut self, stream: &Arc<CudaStream>, experiment: &Experiment) {
        self.pack_kernel(stream, &experiment.quantize);
    }

    fn pack_kernel(&mut self, stream: &Arc<CudaStream>, kernel: &CudaFunction) {
        let groups = self.rows * self.inputs / 32;
        let input = self
            .input_gpu
            .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
        let mut scales = self
            .scales_gpu
            .slice_mut(SCALE_PREFIX..SCALE_PREFIX + groups);
        let mut words = self
            .words_gpu
            .slice_mut(WORD_PREFIX..WORD_PREFIX + groups * 8);
        let mut launch = stream.launch_builder(kernel);
        let (rows, inputs) = (self.rows as u32, self.inputs as u32);
        launch
            .arg(&input)
            .arg(&mut scales)
            .arg(&mut words)
            .arg(&rows)
            .arg(&inputs);
        // SAFETY: retained pack ABI, one complete warp per K32 group. Excess
        // final warps exit before writing; all typed views retain their extents.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: ((groups as u32).div_ceil(4), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    fn multiply(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        experiment: &Experiment,
        route: Route,
        tile: u32,
    ) {
        let approximate = !matches!(route, Route::Baseline);
        let groups = self.rows * self.inputs / 32;
        let kernel = if approximate {
            experiment.dot(self.format, tile, route.integer_partial_values().unwrap())
        } else {
            match (self.format, tile) {
                (GgufBlockFormat::Q4K, 1) => &kernels.linear_q4k_f16,
                (GgufBlockFormat::Q4K, 8) => &kernels.linear_q4k_tiled_f16,
                (GgufBlockFormat::Q5K, 1) => &kernels.linear_q5k_f16,
                (GgufBlockFormat::Q5K, 8) => &kernels.linear_q5k_tiled_f16,
                (GgufBlockFormat::Iq4Xs, 1) => &kernels.linear_iq4xs_f16,
                (GgufBlockFormat::Iq4Xs, 8) => &kernels.linear_iq4xs_tiled_f16,
                _ => panic!("unsupported control route"),
            }
        };
        let input = self
            .input_gpu
            .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
        let weights = self
            .weights_gpu
            .slice(WEIGHT_PREFIX..self.weights.len() - 7);
        let scales = self.scales_gpu.slice(SCALE_PREFIX..SCALE_PREFIX + groups);
        let words = self.words_gpu.slice(WORD_PREFIX..WORD_PREFIX + groups * 8);
        let mut output = self
            .output_gpu
            .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + self.rows * self.stride);
        let params = [
            self.rows as u32,
            self.inputs as u32,
            self.outputs as u32,
            self.stride as u32,
            COLUMN_OFFSET as u32,
            self.format.ggml_type_id(),
            256,
            self.format.block_bytes() as u32,
        ];
        let mut launch = stream.launch_builder(kernel);
        if approximate {
            launch.arg(&scales).arg(&words);
        } else {
            launch.arg(&input);
        }
        launch.arg(&weights).arg(&mut output);
        for p in &params[..if approximate { 5 } else { 8 }] {
            launch.arg(p);
        }
        // SAFETY: original byte-safe weight view at offset5, full K256 blocks,
        // complete output spans, four warps/CTA, guarded column and row tails.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (self.outputs as u32).div_ceil(4),
                    (self.rows as u32).div_ceil(tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    pub(super) fn run(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        experiment: &Experiment,
        route: Route,
        tile: u32,
        iterations: u32,
    ) -> (f64, f64) {
        assert!(iterations > 0 && self.inputs.is_multiple_of(256));
        self.reset(stream);
        if !route.includes_quantization() {
            self.pack(stream, experiment);
        }
        stream.synchronize().unwrap();
        let wall = std::time::Instant::now();
        let start = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        for _ in 0..iterations {
            if route.includes_quantization() {
                self.pack(stream, experiment);
            }
            self.multiply(stream, kernels, experiment, route, tile);
        }
        let end = stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap();
        end.synchronize().unwrap();
        (
            wall.elapsed().as_secs_f64() * 1e9,
            f64::from(start.elapsed_ms(&end).unwrap()) * 1e6,
        )
    }

    fn validate_pack(&self, stream: &Arc<CudaStream>) {
        self.validate_pack_layout(stream, QwordLayout::GroupMajor);
    }

    fn validate_pack_layout(&self, stream: &Arc<CudaStream>, layout: QwordLayout) {
        let expected = pack_rows(
            &self.input[INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs],
            self.rows,
            self.inputs,
        );
        for (i, &value) in stream
            .clone_dtoh(&self.scales_gpu)
            .unwrap()
            .iter()
            .enumerate()
        {
            if let Some(local) = i
                .checked_sub(SCALE_PREFIX)
                .filter(|&j| j < expected.scales.len())
            {
                let want = expected.scales[local];
                assert!(
                    if want.is_nan() {
                        value.is_nan()
                    } else {
                        value.to_bits() == want.to_bits()
                    },
                    "activation scale {local}"
                );
            } else {
                assert_eq!(value.to_bits(), (-8765.25_f32).to_bits(), "scale guard {i}");
            }
        }
        for (i, &value) in stream
            .clone_dtoh(&self.words_gpu)
            .unwrap()
            .iter()
            .enumerate()
        {
            if let Some(local) = i
                .checked_sub(WORD_PREFIX)
                .filter(|&j| j < expected.quants.len() / 4)
            {
                let logical = layout.logical(local, self.inputs / 32);
                let q = &expected.quants[logical * 4..logical * 4 + 4];
                assert_eq!(
                    value,
                    u32::from_le_bytes([q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8]),
                    "activation word {local}"
                );
            } else {
                assert_eq!(value, 0xaabbccdd, "word guard {i}");
            }
        }
        assert!(stream
            .clone_dtoh(&self.input_gpu)
            .unwrap()
            .iter()
            .zip(&self.input)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        assert_eq!(stream.clone_dtoh(&self.weights_gpu).unwrap(), self.weights);
    }

    pub(super) fn pack_probe(
        stream: &Arc<CudaStream>,
        experiment: &Experiment,
        rows: usize,
        inputs: usize,
        values: &[f16],
    ) {
        let mut case = Self::new(stream, GgufBlockFormat::Q4K, rows, inputs, 1);
        case.input[INPUT_PREFIX..INPUT_PREFIX + rows * inputs].copy_from_slice(values);
        stream
            .memcpy_htod(&case.input, &mut case.input_gpu)
            .unwrap();
        for _ in 0..2 {
            case.reset(stream);
            case.pack(stream, experiment);
            case.validate_pack(stream);
        }
    }

    // Compare the same activation row and weights across batch sizes, not two
    // independently generated matrices. Other M8 rows retain different inputs.
    pub(super) fn batch_row_equivalence(
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        experiment: &Experiment,
        format: GgufBlockFormat,
        inputs: usize,
        outputs: usize,
    ) {
        let mut single = Self::new(stream, format, 1, inputs, outputs);
        let mut batch = Self::new(stream, format, 8, inputs, outputs);
        assert_eq!(single.weights, batch.weights);
        let input = &single.input[INPUT_PREFIX..][..inputs];
        let positions = [0, 4, 7];
        for row in positions {
            batch.input[INPUT_PREFIX + row * inputs..][..inputs].copy_from_slice(input);
        }
        assert!(batch.input[INPUT_PREFIX + inputs..][..inputs]
            .iter()
            .zip(input)
            .any(|(a, b)| a.to_bits() != b.to_bits()));
        stream
            .memcpy_htod(&batch.input, &mut batch.input_gpu)
            .unwrap();
        batch.prepare_references();
        // Each association has its own reference. Dot4 and complete-group32
        // share a mathematical policy, but their FP32 output bits may differ.
        let route_groups = if format == GgufBlockFormat::Q5K {
            &Route::QUANTIZED_GROUPS[1..]
        } else {
            &Route::QUANTIZED_GROUPS[..]
        };
        for &routes in route_groups {
            let mut reference = None;
            for (case, selected_rows) in [(&mut single, &[0][..]), (&mut batch, &positions[..])] {
                for tile in [1, 8] {
                    for route in routes {
                        for _ in 0..2 {
                            case.run(stream, kernels, experiment, route, tile, 1);
                            // Full oracle, guard, and immutable-input checks run
                            // before extracting only the matching row's evidence.
                            let (output, _) = case.validate(stream, route);
                            let words = stream.clone_dtoh(&case.words_gpu).unwrap();
                            let scales = stream.clone_dtoh(&case.scales_gpu).unwrap();
                            for &row in selected_rows {
                                let actual = (
                                    words[WORD_PREFIX + row * inputs / 4..][..inputs / 4].to_vec(),
                                    scales[SCALE_PREFIX + row * inputs / 32..][..inputs / 32]
                                        .iter()
                                        .map(|x| x.to_bits())
                                        .collect::<Vec<_>>(),
                                    output[OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET..]
                                        [..outputs]
                                        .to_vec(),
                                );
                                if let Some((q, d, y)) = &reference {
                                    assert_eq!(
                                        &actual.0, q,
                                        "{format:?} M{} row{row} T{tile} {route:?} q",
                                        case.rows
                                    );
                                    assert_eq!(
                                        &actual.1, d,
                                        "{format:?} M{} row{row} T{tile} {route:?} scale",
                                        case.rows
                                    );
                                    assert_eq!(
                                        &actual.2, y,
                                        "{format:?} M{} row{row} T{tile} {route:?} output",
                                        case.rows
                                    );
                                } else {
                                    reference = Some(actual);
                                }
                            }
                        }
                    }
                }
            }
            println!(
                "{}",
                serde_json::json!({
                    "experiment":"activation_q8_batch_row_equivalence",
                    "format":format!("{format:?}"), "input":inputs, "output":outputs,
                    "batch_rows":[1,8], "m8_positions":positions, "row_tiles":[1,8],
                    "routes":routes.map(Route::name),
                    "integer_partial_values":routes[0].integer_partial_values(),
                    "evidence":"packed qwords, F32 scale bits, F16 output bits"
                })
            );
        }
    }

    // Preserve the historical min-only / exact-cancellation regressions, and
    // add signed/zero/subnormal outer scales for the new IQ4 route.
    pub(super) fn edge_probes(
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        experiment: &Experiment,
    ) {
        for (q, minimum) in [(0, 1), (15, 15)] {
            let mut case = Self::new(stream, GgufBlockFormat::Q4K, 3, 256, 7);
            let block = q4k_reference::Q4Block {
                d: f16::ONE,
                dmin: f16::ONE,
                scales: [1; 8],
                minima: [minimum; 8],
                quants: [q; 256],
            };
            for row in 0..case.rows {
                let input = &mut case.input[INPUT_PREFIX + row * 256..][..256];
                input.fill(f16::ZERO);
                input[0] = f16::ONE;
                input[1] = f16::from_f32(1.0 / 256.0);
            }
            for col in 0..case.outputs {
                case.weights[WEIGHT_PREFIX + col * 144..][..144].copy_from_slice(&block.encode());
            }
            case.check_modified(stream, kernels, experiment);
            if q == 15 {
                let output = stream.clone_dtoh(&case.output_gpu).unwrap();
                for row in 0..case.rows {
                    for value in
                        &output[OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET..][..case.outputs]
                    {
                        assert_eq!(value.to_f32(), 0.0, "exact Q4 affine cancellation");
                    }
                }
            }
        }
        let mut iq = Self::new(stream, GgufBlockFormat::Iq4Xs, 3, 256, 9);
        for (col, bits) in [
            0_u16, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0xac00,
        ]
        .into_iter()
        .enumerate()
        {
            iq.weights[WEIGHT_PREFIX + col * 136..][..2].copy_from_slice(&bits.to_le_bytes());
        }
        iq.check_modified(stream, kernels, experiment);

        for format in [GgufBlockFormat::Q4K, GgufBlockFormat::Iq4Xs] {
            let mut case = Self::new(stream, format, 1, 256, 7);
            case.input[INPUT_PREFIX] = f16::NAN;
            case.input[INPUT_PREFIX + 32] = f16::INFINITY;
            case.input[INPUT_PREFIX + 64] = f16::NEG_INFINITY;
            stream
                .memcpy_htod(&case.input, &mut case.input_gpu)
                .unwrap();
            for tile in [1, 8] {
                for route in [Route::DotOnly, Route::Group32DotOnly] {
                    for _ in 0..2 {
                        case.reset(stream);
                        // A finite poison distinguishes an actual NaN write from
                        // an unwritten destination in this exceptional-value case.
                        let mut initial =
                            vec![f16::from_f32(-12345.0); OUTPUT_PREFIX + case.stride + 5];
                        initial[OUTPUT_PREFIX + COLUMN_OFFSET..][..case.outputs].fill(f16::ZERO);
                        stream.memcpy_htod(&initial, &mut case.output_gpu).unwrap();
                        case.pack(stream, experiment);
                        case.multiply(stream, kernels, experiment, route, tile);
                        case.validate_pack(stream);
                        let output = stream.clone_dtoh(&case.output_gpu).unwrap();
                        for (i, value) in output.iter().enumerate() {
                            if (OUTPUT_PREFIX + COLUMN_OFFSET
                                ..OUTPUT_PREFIX + COLUMN_OFFSET + case.outputs)
                                .contains(&i)
                            {
                                assert!(
                                    value.is_nan(),
                                    "nonfinite activation did not propagate for {format:?}"
                                );
                            } else {
                                assert_eq!(
                                    value.to_bits(),
                                    initial[i].to_bits(),
                                    "exceptional output guard {i}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_modified(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        experiment: &Experiment,
    ) {
        stream
            .memcpy_htod(&self.input, &mut self.input_gpu)
            .unwrap();
        stream
            .memcpy_htod(&self.weights, &mut self.weights_gpu)
            .unwrap();
        self.prepare_references();
        for tile in [1, 8] {
            for route in Route::ALL {
                let mut stable = None;
                for _ in 0..2 {
                    self.run(stream, kernels, experiment, route, tile, 1);
                    let (bits, _) = self.validate(stream, route);
                    if let Some(previous) = &stable {
                        assert_eq!(&bits, previous);
                    } else {
                        stable = Some(bits);
                    }
                }
            }
        }
    }

    pub(super) fn validate(
        &self,
        stream: &Arc<CudaStream>,
        route: Route,
    ) -> (Vec<u16>, serde_json::Value) {
        self.validate_pack(stream);
        self.validate_output(stream, route)
    }

    fn validate_output(
        &self,
        stream: &Arc<CudaStream>,
        route: Route,
    ) -> (Vec<u16>, serde_json::Value) {
        let actual = stream.clone_dtoh(&self.output_gpu).unwrap();
        let approximate = !matches!(route, Route::Baseline);
        let mut errors = [0.0_f64; 5];
        let mut samples = 0;
        for (i, &value) in actual.iter().enumerate() {
            let position = i.checked_sub(OUTPUT_PREFIX).filter(|&j| {
                j < self.rows * self.stride
                    && (COLUMN_OFFSET..COLUMN_OFFSET + self.outputs).contains(&(j % self.stride))
            });
            let Some(position) = position else {
                assert_eq!(
                    value.to_bits(),
                    f16::from_f32(-12345.0).to_bits(),
                    "output guard {i}"
                );
                continue;
            };
            assert!(value.is_finite(), "unwritten/nonfinite output {i}");
            let Some(reference) = self.references
                [position / self.stride * self.outputs + position % self.stride - COLUMN_OFFSET]
            else {
                continue;
            };
            samples += 1;
            let nu = if let Some(partial_values) = route.integer_partial_values() {
                // R1 has 4 groups/warp and 8 dot4 lanes/group. R2 gives each
                // lane one complete group and advances by 32 groups. Integer
                // partials are exact; bound at most 4 rescale operations, the
                // per-lane serial sum and 5 final warp-tree additions. Expanded
                // magnitudes retain the Q4 min terms before cancellation.
                let groups_per_serial_step = if partial_values == 32 { 32 } else { 4 };
                ((self.inputs / 32).div_ceil(groups_per_serial_step) + 9) as f64
            } else {
                self.inputs as f64
            } * f64::from(f32::EPSILON);
            let magnitude = if approximate {
                reference.expanded_abs
            } else {
                reference.original_abs
            };
            let expected = if approximate {
                reference.policy
            } else {
                reference.original
            };
            let accumulation_bound = nu / (1.0 - nu) * magnitude;
            let bound = accumulation_bound
                + <f16 as Scalar>::ROUNDING * (expected.abs() + accumulation_bound)
                + f64::from(f16::from_bits(1).to_f32()) / 2.0;
            let observed = f64::from(value.to_f32());
            let error = (observed - expected).abs();
            assert!(error <= bound,"{route:?} output {i}: {observed} expected {expected}, error {error}, bound {bound}");
            errors[0] = errors[0].max(error);
            errors[1] = errors[1].max((reference.quantized - reference.original).abs());
            errors[2] = errors[2].max((reference.policy - reference.quantized).abs());
            errors[3] = errors[3].max((observed - reference.quantized).abs());
            errors[4] = errors[4].max((observed - reference.original).abs());
            let f64_slack = self.inputs as f64
                * f64::EPSILON
                * (reference.original_abs + reference.expanded_abs);
            assert!(
                (reference.quantized - reference.original).abs()
                    <= reference.activation_bound + f64_slack
            );
        }
        (
            actual.iter().map(|x| x.to_bits()).collect(),
            serde_json::json!({
                "oracle_samples":samples,"output_elements":self.rows*self.outputs,
                "gpu_vs_route_oracle_max_abs":errors[0],"activation_quantization_max_abs":errors[1],
                "factored_policy_vs_decoded_q8_max_abs":errors[2],
                "gpu_vs_decoded_q8_max_abs":errors[3],"gpu_vs_original_max_abs":errors[4],
                "oracle_scope":if self.outputs<=49 {"all outputs"} else {"first/middle/last rows and columns; all outputs checked finite/stable"}
            }),
        )
    }
}
