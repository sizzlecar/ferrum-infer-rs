//! Ignored primitive tests for the explicitly approximate Q4_K/Q5_K/Q6_K × Q8 policy.
//! Model composition, output quality and service performance are separate checks.

use super::*;
use crate::gguf_blocks::q4k_q8_reference::{
    dot_reference, fixture_block, pack_rows, DotReference, Q4Block,
};
use crate::gguf_blocks::q56k_q8_reference::{
    dot_q5, dot_q6, fixture_q5, fixture_q6, Q5Block, Q6Block,
};
use cudarc::driver::{
    sys::{CUdevice_attribute, CUevent_flags},
    CudaSlice,
};

const INPUT_PREFIX: usize = 3;
const WEIGHT_PREFIX: usize = 5;
const SCALE_PREFIX: usize = 2;
const WORD_PREFIX: usize = 3;
const OUTPUT_PREFIX: usize = 4;
const COLUMN_OFFSET: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Q4,
    Q5,
    Q6,
}

impl Format {
    fn symbol_stem(self) -> &'static str {
        match self {
            Self::Q4 => "q4k",
            Self::Q5 => "q5k",
            Self::Q6 => "q6k",
        }
    }

    fn parameters(self) -> (u32, usize) {
        match self {
            Self::Q4 => (12, 144),
            Self::Q5 => (13, 176),
            Self::Q6 => (14, 210),
        }
    }

    fn fixture(self, column: usize, block: usize) -> Block {
        match self {
            Self::Q4 => Block::Q4(fixture_block(column, block)),
            Self::Q5 => Block::Q5(fixture_q5(column, block)),
            Self::Q6 => Block::Q6(fixture_q6(column, block)),
        }
    }

    fn reference(
        self,
        input: &[f16],
        column: usize,
        block_override: Option<&Block>,
    ) -> DotReference {
        macro_rules! reference {
            ($variant:ident, $fixture:ident, $dot:ident) => {{
                let blocks: Vec<_> = (0..input.len() / 256)
                    .map(|index| match block_override {
                        Some(Block::$variant(block)) => block.clone(),
                        Some(_) => panic!("weight override format does not match {self:?}"),
                        None => $fixture(column, index),
                    })
                    .collect();
                $dot(input, &blocks)
            }};
        }
        match self {
            Self::Q4 => reference!(Q4, fixture_block, dot_reference),
            Self::Q5 => reference!(Q5, fixture_q5, dot_q5),
            Self::Q6 => reference!(Q6, fixture_q6, dot_q6),
        }
    }
}

enum Block {
    Q4(Q4Block),
    Q5(Q5Block),
    Q6(Q6Block),
}

impl Block {
    fn encode(&self) -> Vec<u8> {
        match self {
            Self::Q4(block) => block.encode().to_vec(),
            Self::Q5(block) => block.encode().to_vec(),
            Self::Q6(block) => block.encode().to_vec(),
        }
    }

    fn format(&self) -> Format {
        match self {
            Self::Q4(_) => Format::Q4,
            Self::Q5(_) => Format::Q5,
            Self::Q6(_) => Format::Q6,
        }
    }
}

struct Prototype {
    pack: CudaFunction,
    lane_scalar: CudaFunction,
    lane_tiled: CudaFunction,
    mma: CudaFunction,
}

impl Prototype {
    fn load(context: &Arc<CudaContext>, format: Format) -> Self {
        let major = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .expect("QK Q8 prototypes require a readable CUDA compute capability");
        assert!(
            major >= 8,
            "QK Q8 MMA prototype requires compute capability >= 8.0; device major={major}"
        );
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .unwrap();
        let load = |suffix| {
            module
                .load_function(&format!(
                    "vnext_gguf_{}_q8_f32scale_{suffix}_f16_prototype",
                    format.symbol_stem()
                ))
                .unwrap()
        };
        Self {
            pack: module
                .load_function("vnext_gguf_q8_f32scale_pack_f16_prototype")
                .unwrap(),
            lane_scalar: load("dp4a_lane"),
            lane_tiled: load("dp4a_lane_tiled"),
            mma: load("mma"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mapping {
    Strict,
    StrictShared,
    Lane,
    Mma,
}

impl Mapping {
    fn approximate(self) -> bool {
        matches!(self, Self::Lane | Self::Mma)
    }
}

struct Case {
    format: Format,
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
    block_override: Option<Block>,
}

impl Case {
    fn new(stream: &Arc<CudaStream>, rows: usize, inputs: usize, outputs: usize) -> Self {
        Self::with_format(stream, Format::Q4, rows, inputs, outputs)
    }

    fn with_format(
        stream: &Arc<CudaStream>,
        format: Format,
        rows: usize,
        inputs: usize,
        outputs: usize,
    ) -> Self {
        let mut input = vec![f16::from_f32(-12345.0); INPUT_PREFIX + rows * inputs + 5];
        for row in 0..rows {
            for col in 0..inputs {
                // Distinct rows and group magnitudes; exact zero groups exercise
                // the d=0 branch without relying on preinitialized device data.
                let x = if (col / 32 + row) % 7 == 0 {
                    0.0
                } else {
                    ((col * 13 + row * 17) % 67) as f32 / 127.0 - 0.25
                };
                input[INPUT_PREFIX + row * inputs + col] = f16::from_f32(x);
            }
        }
        let mut weights = vec![0xcc; WEIGHT_PREFIX];
        for col in 0..outputs {
            for block in 0..inputs / 256 {
                weights.extend(format.fixture(col, block).encode());
            }
        }
        weights.extend([0xcc; 7]);
        let groups = rows * inputs / 32;
        let stride = outputs + 5;
        Self {
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
            block_override: None,
        }
    }

    fn replace_block(&mut self, stream: &Arc<CudaStream>, block: Block) {
        assert_eq!(block.format(), self.format);
        let encoded = block.encode();
        let end = self.weights.len() - 7;
        for slot in self.weights[WEIGHT_PREFIX..end].chunks_exact_mut(self.format.parameters().1) {
            slot.copy_from_slice(&encoded);
        }
        self.block_override = Some(block);
        stream
            .memcpy_htod(&self.weights, &mut self.weights_gpu)
            .unwrap();
    }

    fn reset_destinations(&mut self, stream: &Arc<CudaStream>, output_poison: f16) {
        let groups = self.rows * self.inputs / 32;
        let mut output = vec![f16::from_f32(-12345.0); OUTPUT_PREFIX + self.rows * self.stride + 5];
        for row in 0..self.rows {
            let start = OUTPUT_PREFIX + row * self.stride + COLUMN_OFFSET;
            output[start..start + self.outputs].fill(output_poison);
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

    fn pack(&mut self, stream: &Arc<CudaStream>, prototype: &Prototype) {
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
        let mut launch = stream.launch_builder(&prototype.pack);
        let (rows, inputs) = (self.rows as u32, self.inputs as u32);
        launch
            .arg(&input)
            .arg(&mut scales)
            .arg(&mut words)
            .arg(&rows)
            .arg(&inputs);
        // SAFETY: one warp per complete group; typed views retain all source
        // and staging bytes. Partial final blocks still contain complete warps.
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
        prototype: &Prototype,
        retained: &CudaNativeBlockKernels,
        mapping: Mapping,
    ) {
        let approximate = mapping.approximate();
        let groups = self.rows * self.inputs / 32;
        let tile = match mapping {
            Mapping::StrictShared => 64,
            Mapping::Mma => 32,
            _ if self.rows == 1 => 1,
            _ => 8,
        };
        let column_tile = match mapping {
            Mapping::StrictShared => 64,
            Mapping::Mma => 16,
            _ => 4,
        };
        let kernel = match (mapping, tile) {
            (Mapping::StrictShared, _) => match self.format {
                Format::Q4 => &retained.gemm_q4k_f16,
                Format::Q5 => &retained.gemm_q5k_f16,
                Format::Q6 => &retained.gemm_q6k_f16,
            },
            (Mapping::Lane, 1) => &prototype.lane_scalar,
            (Mapping::Lane, _) => &prototype.lane_tiled,
            (Mapping::Mma, _) => &prototype.mma,
            (Mapping::Strict, 1) if self.format == Format::Q4 => &retained.linear_q4k_f16,
            (Mapping::Strict, _) if self.format == Format::Q4 => &retained.linear_q4k_tiled_f16,
            (Mapping::Strict, 1) => &retained.linear_f16,
            (Mapping::Strict, _) => &retained.linear_tiled_f16,
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
            self.format.parameters().0,
            256,
            self.format.parameters().1 as u32,
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
        // SAFETY: complete native blocks may be byte-unaligned;
        // kernels use byte loads. Output interval is inside every row stride.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (self.outputs as u32).div_ceil(column_tile),
                    (self.rows as u32).div_ceil(tile),
                    1,
                ),
                block_dim: if mapping == Mapping::StrictShared {
                    (16, 16, 1)
                } else {
                    (128, 1, 1)
                },
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    fn validate_pack(&self, stream: &Arc<CudaStream>) {
        let source = &self.input[INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs];
        let expected = pack_rows(source, self.rows, self.inputs);
        let scales = stream.clone_dtoh(&self.scales_gpu).unwrap();
        for (i, &d) in scales.iter().enumerate() {
            if let Some(local) = i
                .checked_sub(SCALE_PREFIX)
                .filter(|&j| j < expected.scales.len())
            {
                let want = expected.scales[local];
                assert!(
                    if want.is_nan() {
                        d.is_nan()
                    } else {
                        d.to_bits() == want.to_bits()
                    },
                    "scale[{local}] {d} != {want}"
                );
            } else {
                assert_eq!(d.to_bits(), (-8765.25_f32).to_bits());
            }
        }
        let words = stream.clone_dtoh(&self.words_gpu).unwrap();
        for (i, &word) in words.iter().enumerate() {
            if let Some(local) = i
                .checked_sub(WORD_PREFIX)
                .filter(|&j| j < expected.quants.len() / 4)
            {
                let q = &expected.quants[local * 4..][..4];
                assert_eq!(
                    word,
                    u32::from_le_bytes([q[0] as u8, q[1] as u8, q[2] as u8, q[3] as u8]),
                    "qword[{local}]"
                );
            } else {
                assert_eq!(word, 0xaabbccdd);
            }
        }
    }

    fn validate_output(
        &self,
        stream: &Arc<CudaStream>,
        approximate: bool,
        all: bool,
    ) -> (f64, f64, f64) {
        let actual = stream.clone_dtoh(&self.output_gpu).unwrap();
        let mut errors = (0.0_f64, 0.0_f64, 0.0_f64);
        for (i, &value) in actual.iter().enumerate() {
            let position = i.checked_sub(OUTPUT_PREFIX).filter(|&j| {
                j < self.rows * self.stride
                    && (COLUMN_OFFSET..COLUMN_OFFSET + self.outputs).contains(&(j % self.stride))
            });
            let Some(position) = position else {
                assert_eq!(
                    value.to_bits(),
                    f16::from_f32(-12345.0).to_bits(),
                    "output guard[{i}]"
                );
                continue;
            };
            let row = position / self.stride;
            let col = position % self.stride - COLUMN_OFFSET;
            if !all
                && (![0, self.rows / 2, self.rows - 1].contains(&row)
                    || ![0, self.outputs / 2, self.outputs - 1].contains(&col))
            {
                continue;
            }
            let reference = self.format.reference(
                &self.input[INPUT_PREFIX + row * self.inputs..][..self.inputs],
                col,
                self.block_override.as_ref(),
            );
            let half_subnormal_rounding = f64::from(f16::from_bits(1).to_f32()) / 2.0;
            let (expected, bound) = if approximate {
                // Four rescale/subtraction operations, at most groups/4
                // serial sums per active lane, then five warp-shuffle sums.
                // Q4/Q5 a/b and Q6 a0/a1 are exact: finite F16 × u6/i8
                // needs at most 18 bits. Q6 shares delta across both K16 dots.
                let nu = ((self.inputs / 32).div_ceil(4) + 9) as f64 * f64::from(f32::EPSILON);
                let accumulation_bound = nu / (1.0 - nu) * reference.expanded_abs_terms;
                (
                    reference.policy,
                    accumulation_bound
                        + <f16 as Scalar>::ROUNDING * (reference.policy.abs() + accumulation_bound)
                        + half_subnormal_rounding,
                )
            } else {
                (
                    reference.strict_original,
                    (self.inputs as f64 * f64::from(f32::EPSILON) + <f16 as Scalar>::ROUNDING)
                        * reference.strict_abs_terms
                        + half_subnormal_rounding,
                )
            };
            let error = (f64::from(value.to_f32()) - expected).abs();
            assert!(value.is_finite() && error <= bound,
                "format={:?} approx={approximate} row={row} col={col} actual={} expected={expected} error={error} bound={bound}", self.format, value.to_f32());
            errors.0 = errors.0.max(error);
            errors.1 = errors
                .1
                .max((reference.strict_quantized - reference.strict_original).abs());
            errors.2 = errors
                .2
                .max((reference.policy - reference.strict_quantized).abs());
        }
        assert_eq!(stream.clone_dtoh(&self.weights_gpu).unwrap(), self.weights);
        assert!(stream
            .clone_dtoh(&self.input_gpu)
            .unwrap()
            .iter()
            .zip(&self.input)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        errors
    }
}

#[test]
#[ignore = "requires an actual SM80+ CUDA device; approximate policy prototype only"]
fn q4k_q8_prototype_pack_and_matmul_match_policy_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("Q4K Q8 prototype requires CUDA");
    let stream = context.default_stream();
    let prototype = Prototype::load(&context, Format::Q4);
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for (rows, inputs, outputs) in [
        (1, 256, 7),
        (3, 768, 9),
        (7, 256, 15),
        (8, 512, 33),
        (9, 256, 5),
        (15, 512, 16),
        (16, 256, 31),
        (31, 512, 32),
        (32, 768, 17),
        (33, 256, 33),
    ] {
        let mut case = Case::new(&stream, rows, inputs, outputs);
        for mapping in [Mapping::Lane, Mapping::Mma] {
            for _ in 0..2 {
                case.reset_destinations(&stream, f16::NAN);
                case.pack(&stream, &prototype);
                case.multiply(&stream, &prototype, &retained, mapping);
                case.validate_pack(&stream);
                case.validate_output(&stream, true, true);
            }
        }
        case.reset_destinations(&stream, f16::NAN);
        case.multiply(&stream, &prototype, &retained, Mapping::Strict);
        case.validate_output(&stream, false, true);
    }
    // The pack ABI permits a final partial block of complete 32-value groups.
    let mut tail_pack = Case::new(&stream, 3, 32, 1);
    tail_pack.reset_destinations(&stream, f16::NAN);
    tail_pack.pack(&stream, &prototype);
    tail_pack.validate_pack(&stream);
    // Min-only policy separation and an exactly cancelling affine weight are
    // not covered by a generic positive-magnitude tolerance fixture.
    for (q, minimum) in [(0, 1), (15, 15)] {
        let mut special = Case::new(&stream, 1, 256, 7);
        let block = Q4Block {
            d: f16::ONE,
            dmin: f16::ONE,
            scales: [1; 8],
            minima: [minimum; 8],
            quants: [q; 256],
        };
        special.input[INPUT_PREFIX..INPUT_PREFIX + 256].fill(f16::ZERO);
        special.input[INPUT_PREFIX] = f16::ONE;
        special.input[INPUT_PREFIX + 1] = f16::from_f32(1.0 / 256.0);
        special.replace_block(&stream, Block::Q4(block));
        stream
            .memcpy_htod(&special.input, &mut special.input_gpu)
            .unwrap();
        for mapping in [Mapping::Lane, Mapping::Mma] {
            special.reset_destinations(&stream, f16::NAN);
            special.pack(&stream, &prototype);
            special.validate_pack(&stream);
            special.multiply(&stream, &prototype, &retained, mapping);
            special.validate_output(&stream, true, true);
            if q == 15 && minimum == 15 {
                let output = stream.clone_dtoh(&special.output_gpu).unwrap();
                for value in &output[OUTPUT_PREFIX + COLUMN_OFFSET..][..special.outputs] {
                    assert_eq!(
                        value.to_f32(),
                        0.0,
                        "exact affine cancellation for {mapping:?}"
                    );
                }
            }
        }
    }
    check_nonfinite(&stream, &prototype, &retained, Format::Q4);
}

fn check_nonfinite(
    stream: &Arc<CudaStream>,
    prototype: &Prototype,
    retained: &CudaNativeBlockKernels,
    format: Format,
) {
    let mut exceptional = Case::with_format(stream, format, 1, 256, 7);
    for (i, x) in [127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
        .into_iter()
        .enumerate()
    {
        exceptional.input[INPUT_PREFIX + i] = f16::from_f32(x);
    }
    exceptional.input[INPUT_PREFIX + 32..INPUT_PREFIX + 64].fill(f16::NEG_ZERO);
    exceptional.input[INPUT_PREFIX + 64..INPUT_PREFIX + 96].fill(f16::from_bits(1));
    exceptional.input[INPUT_PREFIX + 96] = f16::MAX;
    exceptional.input[INPUT_PREFIX + 97] = f16::MIN;
    exceptional.input[INPUT_PREFIX + 128] = f16::NAN;
    exceptional.input[INPUT_PREFIX + 160] = f16::INFINITY;
    exceptional.input[INPUT_PREFIX + 192] = f16::NEG_INFINITY;
    stream
        .memcpy_htod(&exceptional.input, &mut exceptional.input_gpu)
        .unwrap();
    for mapping in [Mapping::Lane, Mapping::Mma] {
        // Expected outputs are NaN here, so a finite poison is necessary to
        // distinguish a real nonfinite write from a skipped output store.
        exceptional.reset_destinations(stream, f16::ZERO);
        exceptional.pack(stream, prototype);
        exceptional.validate_pack(stream);
        exceptional.multiply(stream, prototype, retained, mapping);
        let output = stream.clone_dtoh(&exceptional.output_gpu).unwrap();
        assert!(
            output[OUTPUT_PREFIX + COLUMN_OFFSET..][..exceptional.outputs]
                .iter()
                .all(|x| x.is_nan())
        );
        let valid =
            OUTPUT_PREFIX + COLUMN_OFFSET..OUTPUT_PREFIX + COLUMN_OFFSET + exceptional.outputs;
        for (index, value) in output.iter().enumerate() {
            if !valid.contains(&index) {
                assert_eq!(value.to_bits(), f16::from_f32(-12345.0).to_bits());
            }
        }
        assert_eq!(
            stream.clone_dtoh(&exceptional.weights_gpu).unwrap(),
            exceptional.weights
        );
        assert!(stream
            .clone_dtoh(&exceptional.input_gpu)
            .unwrap()
            .iter()
            .zip(&exceptional.input)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }
}

fn check_case(
    case: &mut Case,
    stream: &Arc<CudaStream>,
    prototype: &Prototype,
    retained: &CudaNativeBlockKernels,
    exact: Option<(f32, f32)>,
) {
    for mapping in [Mapping::Strict, Mapping::Lane, Mapping::Mma] {
        for _ in 0..2 {
            case.reset_destinations(stream, f16::NAN);
            let approximate = mapping.approximate();
            if approximate {
                case.pack(stream, prototype);
            }
            case.multiply(stream, prototype, retained, mapping);
            if approximate {
                case.validate_pack(stream);
            }
            case.validate_output(stream, approximate, true);
            if let Some((strict, policy)) = exact {
                let output = stream.clone_dtoh(&case.output_gpu).unwrap();
                for row in 0..case.rows {
                    let start = OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET;
                    for value in &output[start..start + case.outputs] {
                        assert_eq!(
                            value.to_f32(),
                            if approximate { policy } else { strict },
                            "exact {:?} value for {mapping:?} row={row}",
                            case.format
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an actual SM80+ CUDA device; approximate policy prototype only"]
fn q56k_q8_prototype_pack_and_matmul_match_policy_oracle_on_cuda() {
    let context = CudaContext::new(0).expect("Q5K/Q6K Q8 prototype requires CUDA");
    let stream = context.default_stream();
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for format in [Format::Q5, Format::Q6] {
        let prototype = Prototype::load(&context, format);
        // Tail rows/columns, both sides of the MMA tile, and multiple K256
        // blocks use distinct per-column/per-block scales and zero Q8 groups.
        for (rows, inputs, outputs) in [
            (1, 256, 7),
            (3, 768, 9),
            (7, 256, 15),
            (8, 512, 33),
            (9, 256, 5),
            (15, 512, 16),
            (16, 256, 31),
            (31, 512, 32),
            (32, 768, 17),
            (33, 256, 33),
        ] {
            let mut case = Case::with_format(&stream, format, rows, inputs, outputs);
            check_case(&mut case, &stream, &prototype, &retained, None);
        }
        match format {
            Format::Q5 => {
                // The high bit participates in exact affine cancellation;
                // min-only also distinguishes quantized sum from original x.
                for (q, minimum, strict, policy) in [(0, 1, -1.00390625, -1.0), (31, 31, 0.0, 0.0)]
                {
                    let mut case = Case::with_format(&stream, format, 1, 256, 7);
                    case.input[INPUT_PREFIX..INPUT_PREFIX + 256].fill(f16::ZERO);
                    case.input[INPUT_PREFIX] = f16::ONE;
                    case.input[INPUT_PREFIX + 1] = f16::from_f32(1.0 / 256.0);
                    stream
                        .memcpy_htod(&case.input, &mut case.input_gpu)
                        .unwrap();
                    case.replace_block(
                        &stream,
                        Block::Q5(Q5Block {
                            low: Q4Block {
                                d: f16::ONE,
                                dmin: f16::ONE,
                                scales: [1; 8],
                                minima: [minimum; 8],
                                quants: [q & 15; 256],
                            },
                            high: [q >= 16; 256],
                        }),
                    );
                    check_case(
                        &mut case,
                        &stream,
                        &prototype,
                        &retained,
                        Some((strict, policy)),
                    );
                }
            }
            Format::Q6 => {
                let mut case = Case::with_format(&stream, format, 1, 256, 7);
                let mut block = Q6Block {
                    d: f16::ONE,
                    scales: [0; 16],
                    quants: [1; 256],
                };
                block.scales[..2].copy_from_slice(&[1, -128]);
                case.input[INPUT_PREFIX..INPUT_PREFIX + 256].fill(f16::ZERO);
                case.input[INPUT_PREFIX] = f16::from_f32(127.0);
                case.input[INPUT_PREFIX + 16] = f16::from_f32(1.0 / 256.0);
                stream
                    .memcpy_htod(&case.input, &mut case.input_gpu)
                    .unwrap();
                case.replace_block(&stream, Block::Q6(block));
                // One delta=1 for the whole K32, so the second K16's input
                // quantizes to zero. Independently packing K16 gives 126.5.
                check_case(
                    &mut case,
                    &stream,
                    &prototype,
                    &retained,
                    Some((126.5, 127.0)),
                );

                case.input[INPUT_PREFIX..INPUT_PREFIX + 256].fill(f16::ONE);
                stream
                    .memcpy_htod(&case.input, &mut case.input_gpu)
                    .unwrap();
                case.replace_block(
                    &stream,
                    Block::Q6(Q6Block {
                        d: f16::ONE,
                        scales: std::array::from_fn(|i| if i % 2 == 0 { 1 } else { -1 }),
                        quants: [31; 256],
                    }),
                );
                // Opposite K16 coefficients cancel before the shared delta
                // multiplication. The expanded-term bound must not hide it.
                check_case(&mut case, &stream, &prototype, &retained, Some((0.0, 0.0)));

                case.input[INPUT_PREFIX..INPUT_PREFIX + 256].fill(f16::from_f32(1.0 / 128.0));
                stream
                    .memcpy_htod(&case.input, &mut case.input_gpu)
                    .unwrap();
                case.replace_block(
                    &stream,
                    Block::Q6(Q6Block {
                        d: f16::ONE,
                        scales: std::array::from_fn(|i| if i % 2 == 0 { -128 } else { 127 }),
                        quants: std::array::from_fn(|i| if (i / 16) % 2 == 0 { -32 } else { 31 }),
                    }),
                );
                check_case(&mut case, &stream, &prototype, &retained, None);
            }
            Format::Q4 => unreachable!(),
        }
        check_nonfinite(&stream, &prototype, &retained, format);
    }
}

#[test]
#[ignore = "paired GPU timing including activation pack; exclusive SM80+ CUDA access"]
fn q4k_q8_prototype_pack_plus_matmul_microbench() {
    // Hot reuse of each compressed matrix, identical cache treatment for both
    // paths. This screens arithmetic/pack cost, not 9B cross-layer throughput.
    let context = CudaContext::new(0).expect("Q4K Q8 timing requires CUDA");
    let stream = context.default_stream();
    let prototype = Prototype::load(&context, Format::Q4);
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for rows in [1, 8, 32] {
        for outputs in [4096, 12288] {
            let mut case = Case::new(&stream, rows, 4096, outputs);
            run_microbench(&mut case, &stream, &prototype, &retained);
        }
    }
}

#[test]
#[ignore = "paired GPU timing including activation pack; exclusive SM80+ CUDA access"]
fn q56k_q8_prototype_pack_plus_matmul_microbench() {
    let context = CudaContext::new(0).expect("Q5K/Q6K Q8 timing requires CUDA");
    let stream = context.default_stream();
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for format in [Format::Q5, Format::Q6] {
        let prototype = Prototype::load(&context, format);
        for rows in [8, 32] {
            let shapes: &[(usize, usize)] = match format {
                Format::Q5 => &[(4096, 4096), (4096, 8192)],
                Format::Q6 => &[(12288, 4096)],
                Format::Q4 => unreachable!(),
            };
            for &(inputs, outputs) in shapes {
                let mut case = Case::with_format(&stream, format, rows, inputs, outputs);
                run_microbench(&mut case, &stream, &prototype, &retained);
            }
        }
    }
}

#[test]
#[ignore = "paired prefill GPU timing including activation pack; exclusive SM80+ CUDA access"]
fn q8_prefill_pack_plus_shared_matmul_microbench() {
    // The strict product uses shared F32 GEMM for these large F16 batches.
    // Decode's original tiled control is not the relevant prefill baseline.
    // These two FFN geometries screen the 9B gate/up and down projections;
    // hot matrix reuse and sampled output oracles do not prove service gains.
    let context = CudaContext::new(0).expect("Q8 prefill timing requires CUDA");
    let stream = context.default_stream();
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for (format, inputs, outputs) in [(Format::Q4, 4096, 12288), (Format::Q6, 12288, 4096)] {
        let prototype = Prototype::load(&context, format);
        for rows in [128, 512] {
            let mut case = Case::with_format(&stream, format, rows, inputs, outputs);
            run_microbench_mappings(
                &mut case,
                &stream,
                &prototype,
                &retained,
                &[Mapping::StrictShared, Mapping::Mma],
            );
        }
    }
}

#[test]
#[ignore = "paired GDN projection timing including activation pack; exclusive SM80+ CUDA access"]
fn native_gdn_projection_mappings_microbench() {
    // QKV, gate and output geometries from the 4B native GGUF inventory.
    // These are individual synthetic projections, not a recurrent-attention
    // implementation or a qualification of Q8 recurrent-state numerics.
    let context = CudaContext::new(0).expect("GDN projection timing requires CUDA");
    let stream = context.default_stream();
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    for (format, inputs, outputs) in [
        (Format::Q5, 2560, 8192),
        (Format::Q4, 2560, 4096),
        (Format::Q5, 4096, 2560),
    ] {
        let prototype = Prototype::load(&context, format);
        for rows in [8, 32, 64, 512] {
            // At 512 rows the existing strict provider already uses shared
            // GEMM. At medium widths, measure both strict mappings before
            // changing their selection; their crossover differs by format.
            let mappings: &[Mapping] = match rows {
                8 => &[Mapping::Strict, Mapping::Mma],
                32 | 64 => &[Mapping::Strict, Mapping::StrictShared, Mapping::Mma],
                _ => &[Mapping::StrictShared, Mapping::Mma],
            };
            let mut case = Case::with_format(&stream, format, rows, inputs, outputs);
            run_microbench_mappings(&mut case, &stream, &prototype, &retained, mappings);
        }
    }
}

#[test]
#[ignore = "paired Q5K projection timing; coordinate exclusive CUDA access"]
fn native_q5k_midrow_dispatch_microbench() {
    let context = CudaContext::new(0).expect("Q5K projection timing requires CUDA");
    let stream = context.default_stream();
    let retained = CudaNativeBlockKernels::load(&context).unwrap();
    let prototype = Prototype::load(&context, Format::Q5);
    for rows in [32, 64] {
        let mut case = Case::with_format(&stream, Format::Q5, rows, 4096, 8192);
        run_microbench_mappings(
            &mut case,
            &stream,
            &prototype,
            &retained,
            &[Mapping::Strict, Mapping::StrictShared],
        );
    }
}

fn run_microbench(
    case: &mut Case,
    stream: &Arc<CudaStream>,
    prototype: &Prototype,
    retained: &CudaNativeBlockKernels,
) {
    run_microbench_mappings(
        case,
        stream,
        prototype,
        retained,
        &[Mapping::Strict, Mapping::Lane, Mapping::Mma],
    );
}

fn run_microbench_mappings(
    case: &mut Case,
    stream: &Arc<CudaStream>,
    prototype: &Prototype,
    retained: &CudaNativeBlockKernels,
    mappings: &[Mapping],
) {
    const REPEATS: u32 = 8;
    for round in 0..6 {
        for index in 0..mappings.len() {
            let mapping = mappings[if round % 2 == 0 {
                index
            } else {
                mappings.len() - 1 - index
            }];
            let approximate = mapping.approximate();
            // Reset outside both GPU and wall timing; REPEATS itself
            // contains only the actual pack (when needed) and GEMV.
            case.reset_destinations(stream, f16::NAN);
            stream.synchronize().unwrap();
            let wall = std::time::Instant::now();
            let start = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            for _ in 0..REPEATS {
                if approximate {
                    case.pack(stream, prototype);
                }
                case.multiply(stream, prototype, retained, mapping);
            }
            let end = stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .unwrap();
            end.synchronize().unwrap();
            let wall_us = wall.elapsed().as_secs_f64() * 1e6 / f64::from(REPEATS);
            let gpu_us = f64::from(start.elapsed_ms(&end).unwrap()) * 1000.0 / f64::from(REPEATS);
            if approximate {
                case.validate_pack(stream);
            }
            let errors = case.validate_output(stream, approximate, false);
            if round > 0 {
                let (rows, inputs, outputs) = (case.rows, case.inputs, case.outputs);
                println!("{}_q8_prototype rows={rows} inputs={inputs} outputs={outputs} round={round} mapping={mapping:?} includes_pack={approximate} gpu_us={gpu_us:.3} wall_us={wall_us:.3} oracle_max_abs={} activation_max_abs={} reconstruction_max_abs={}",case.format.symbol_stem(),errors.0,errors.1,errors.2);
            }
        }
    }
}
