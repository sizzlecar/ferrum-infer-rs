use super::*;

pub(super) const TEMPLATES: usize = 16;
const INPUT_PREFIX: usize = 3;
const WEIGHT_PREFIX: usize = 5; // Deliberately odd, including delta at +208.
const OUTPUT_PREFIX: usize = 8;
const COLUMN_OFFSET: usize = 3;
const TRAILER: usize = 7;

/// Sixteen independent physical column templates, not sixteen sparse columns.
/// A full head has all 248320 contiguous compressed rows but repeats these
/// templates. Only this small decoded oracle is retained, never N*K F32.
pub(super) fn templates(inputs: usize) -> (Vec<u8>, Vec<f32>) {
    assert!(inputs > 0 && inputs % 256 == 0);
    let mut raw = vec![0_u8; TEMPLATES * (inputs / 256) * 210];
    for (i, block) in raw.chunks_exact_mut(210).enumerate() {
        let template = i / (inputs / 256);
        let k_block = i % (inputs / 256);
        for (j, byte) in block[..192].iter_mut().enumerate() {
            *byte = (j * 43 + template * 67 + k_block * 29 + (j / 3) * 11) as u8;
        }
        for (j, byte) in block[192..208].iter_mut().enumerate() {
            let scale = [-128_i8, -127, -17, -1, 0, 1, 16, 127][(j + template + k_block) % 8];
            *byte = scale as u8;
        }
        let delta_bits: u16 = match (template + k_block) % 8 {
            0 => 0,
            1 => 1,
            2 => 0x8001,
            3 => 0x2000,
            4 => 0xa000,
            _ => f16::from_f32((1 + (template + k_block) % 5) as f32 / 4096.0).to_bits(),
        };
        block[208..210].copy_from_slice(&delta_bits.to_le_bytes());
    }
    let mut decoded = vec![0.0; TEMPLATES * inputs];
    GgufBlockFormat::Q6K.decode(&raw, &mut decoded).unwrap();
    (raw, decoded)
}

pub(super) fn input_values<I: Scalar>(rows: usize, inputs: usize, generation: usize) -> Vec<I> {
    let mut result = vec![I::from_f32(-12345.0); INPUT_PREFIX + rows * inputs + TRAILER];
    for row in 0..rows {
        for k in 0..inputs {
            // Dense, bounded, and not generally representable in F16. Each
            // generation changes the actual stored input at the same pointer.
            let value = (((k * 11 + row * 7 + generation * 19) % 71) as f32 - 35.0)
                / (127.31 * (generation + 1) as f32);
            result[INPUT_PREFIX + row * inputs + k] = I::from_f32(value);
        }
    }
    result
}

pub(super) struct Fixture<I: Scalar, O: Scalar> {
    pub rows: usize,
    pub inputs: usize,
    pub outputs: usize,
    stride: usize,
    input_type: ElementType,
    output_type: ElementType,
    input: Vec<I>,
    weights: Vec<u8>,
    decoded_templates: Vec<f32>,
    initial: Vec<O>,
    expected: Vec<(f64, f64)>,
    input_gpu: CudaSlice<I>,
    weights_gpu: CudaSlice<u8>,
    output_gpu: CudaSlice<O>,
}

impl<I: Scalar, O: Scalar> Fixture<I, O> {
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        rows: usize,
        inputs: usize,
        outputs: usize,
        input_type: ElementType,
        output_type: ElementType,
    ) -> Self {
        let (raw_templates, decoded_templates) = templates(inputs);
        let row_bytes = (inputs / 256) * 210;
        let mut weights = Vec::with_capacity(WEIGHT_PREFIX + outputs * row_bytes + TRAILER);
        weights.resize(WEIGHT_PREFIX, 0xcc);
        for col in 0..outputs {
            let start = (col % TEMPLATES) * row_bytes;
            weights.extend_from_slice(&raw_templates[start..start + row_bytes]);
        }
        weights.extend([0xcc; TRAILER]);
        let stride = outputs + COLUMN_OFFSET + 4;
        let initial = vec![O::from_f32(-12345.0); OUTPUT_PREFIX + rows * stride + TRAILER];
        let input = input_values(rows, inputs, 0);
        let mut case = Self {
            rows,
            inputs,
            outputs,
            stride,
            input_type,
            output_type,
            input_gpu: stream.clone_htod(&input).unwrap(),
            weights_gpu: stream.clone_htod(&weights).unwrap(),
            output_gpu: stream.clone_htod(&initial).unwrap(),
            input,
            weights,
            decoded_templates,
            initial,
            expected: Vec::new(),
        };
        case.oracle();
        case
    }

    fn oracle(&mut self) {
        self.expected.clear();
        for row in 0..self.rows {
            for template in self.decoded_templates.chunks_exact(self.inputs) {
                let products = self.input[INPUT_PREFIX + row * self.inputs..][..self.inputs]
                    .iter()
                    .zip(template)
                    .map(|(x, &w)| f64::from(x.as_f32()) * f64::from(w));
                self.expected
                    .push((products.clone().sum(), products.map(f64::abs).sum()));
            }
        }
    }

    pub(super) fn update(&mut self, stream: &Arc<CudaStream>, generation: usize) {
        stream.synchronize().unwrap();
        self.input = input_values(self.rows, self.inputs, generation);
        stream
            .memcpy_htod(&self.input, &mut self.input_gpu)
            .unwrap();
        self.oracle();
        stream.synchronize().unwrap();
    }

    pub(super) fn reset(&mut self, stream: &Arc<CudaStream>) {
        stream
            .memcpy_htod(&self.initial, &mut self.output_gpu)
            .unwrap();
        stream.synchronize().unwrap();
    }

    /// DevicePtr access/event guards are all created and destroyed outside
    /// capture. The owning slices in self must outlive every returned graph;
    /// callers keep the fixture alive and synchronize before updating/reaping.
    pub(super) fn capture(
        &mut self,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
    ) -> CudaGraph {
        stream.synchronize().unwrap();
        let part = weights::MatrixPart {
            component_id: WeightId::new("component.q6k-fixed-abi").unwrap(),
            format: weights::MatrixFormat::Block(GgufBlockFormat::Q6K),
            rows: self.outputs as u32,
            columns: self.inputs as u32,
            output_offset: COLUMN_OFFSET as u32,
            transform: None,
            signs_region: None,
        };
        let input = self
            .input_gpu
            .slice(INPUT_PREFIX..INPUT_PREFIX + self.rows * self.inputs);
        let weights = self
            .weights_gpu
            .slice(WEIGHT_PREFIX..self.weights.len() - TRAILER);
        let mut output = self
            .output_gpu
            .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + self.rows * self.stride);
        let (ip, ig) = input.device_ptr(stream);
        let (wp, wg) = weights.device_ptr(stream);
        let (op, og) = output.device_ptr_mut(stream);
        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .unwrap();
        kernels
            .linear_with_precision(
                stream,
                ip,
                wp,
                op,
                &part,
                self.rows as u32,
                self.stride as u32,
                self.input_type,
                self.output_type,
            )
            .unwrap();
        let graph = stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .unwrap()
            .expect("one real production matrix launch");
        drop((ig, wg, og));
        stream.synchronize().unwrap();
        graph
    }

    pub(super) fn validate(&self, stream: &Arc<CudaStream>) -> Vec<u32> {
        stream.synchronize().unwrap();
        let actual = stream.clone_dtoh(&self.output_gpu).unwrap();
        let mut bits = Vec::with_capacity(actual.len());
        for (index, value) in actual.iter().enumerate() {
            let location = index
                .checked_sub(OUTPUT_PREFIX)
                .filter(|i| *i < self.rows * self.stride)
                .filter(|i| {
                    (COLUMN_OFFSET..COLUMN_OFFSET + self.outputs).contains(&(i % self.stride))
                });
            let observed = value.as_f32();
            if let Some(i) = location {
                let row = i / self.stride;
                let col = i % self.stride - COLUMN_OFFSET;
                let (expected, absolute) = self.expected[row * TEMPLATES + col % TEMPLATES];
                // Same independent GGUF/F64 bound as shared_dispatch::check.
                let bound =
                    (self.inputs as f64 * f64::from(f32::EPSILON) + O::ROUNDING) * absolute + 1e-6;
                assert!(observed.is_finite() && (f64::from(observed) - expected).abs() <= bound,
                    "Q6 {:?}->{:?} {}x{}x{} at {row},{col}: {observed} F64={expected} bound={bound}",
                    self.input_type, self.output_type, self.rows, self.inputs, self.outputs);
            } else {
                assert_eq!(
                    observed.to_bits(),
                    self.initial[index].as_f32().to_bits(),
                    "guard {index}"
                );
            }
            bits.push(observed.to_bits());
        }
        bits
    }

    pub(super) fn immutable(&self, stream: &Arc<CudaStream>) {
        stream.synchronize().unwrap();
        assert!(
            stream
                .clone_dtoh(&self.input_gpu)
                .unwrap()
                .iter()
                .zip(&self.input)
                .all(|(a, b)| a.as_f32().to_bits() == b.as_f32().to_bits()),
            "input mutated"
        );
        assert_eq!(
            stream.clone_dtoh(&self.weights_gpu).unwrap(),
            self.weights,
            "weights mutated"
        );
    }
}

pub(super) fn qualify<I: Scalar, O: Scalar>(
    case: &mut Fixture<I, O>,
    stream: &Arc<CudaStream>,
    old: &CudaNativeBlockKernels,
    candidate: &CudaNativeBlockKernels,
) -> Vec<Vec<u32>> {
    let old_graph = case.capture(stream, old);
    let new_graph = case.capture(stream, candidate);
    let mut references: Vec<Vec<u32>> = Vec::new();
    for generation in 0..2 {
        case.update(stream, generation);
        case.reset(stream);
        old_graph.launch().unwrap();
        let expected = case.validate(stream);
        case.reset(stream);
        new_graph.launch().unwrap();
        let actual = case.validate(stream);
        assert_eq!(
            actual, expected,
            "two-generation generic/product bitwise mismatch"
        );
        if let Some(previous) = references.last() {
            assert_ne!(
                &actual, previous,
                "changed stored input did not change output"
            );
        }
        references.push(expected);
        case.immutable(stream);
    }
    references
}
