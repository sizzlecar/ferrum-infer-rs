//! IQ4/Q4/Q5 G32 layout counterfactual. Only activation qword addressing changes.
//! Resident weights, identical arithmetic and retained exports are controls;
//! this is neither a production route nor evidence about streaming weights.

use super::graph_paired::{balanced_route, measure, Captured};
use super::*;

mod correctness;
mod paired;

#[derive(Clone, Copy, Debug)]
enum LayoutRoute {
    RetainedF32,
    GroupInclusive,
    GroupDotOnly,
    WordInclusive,
    WordDotOnly,
}

impl LayoutRoute {
    const ALL: [Self; 5] = [
        Self::RetainedF32,
        Self::GroupInclusive,
        Self::GroupDotOnly,
        Self::WordInclusive,
        Self::WordDotOnly,
    ];
    const APPROXIMATE: [Self; 4] = [
        Self::GroupInclusive,
        Self::GroupDotOnly,
        Self::WordInclusive,
        Self::WordDotOnly,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::RetainedF32 => "retained_f32",
            Self::GroupInclusive => "group32_group_major_inclusive",
            Self::GroupDotOnly => "group32_group_major_dot_only",
            Self::WordInclusive => "group32_word_major_inclusive",
            Self::WordDotOnly => "group32_word_major_dot_only",
        }
    }

    fn layout(self) -> QwordLayout {
        match self {
            Self::WordInclusive | Self::WordDotOnly => QwordLayout::WordMajor,
            _ => QwordLayout::GroupMajor,
        }
    }

    fn numerical_route(self) -> Route {
        match self {
            Self::RetainedF32 => Route::Baseline,
            Self::GroupInclusive | Self::WordInclusive => Route::Group32QuantizeAndDot,
            Self::GroupDotOnly | Self::WordDotOnly => Route::Group32DotOnly,
        }
    }

    fn includes_pack(self) -> bool {
        self.numerical_route().includes_quantization()
    }
}

struct LayoutExperiment {
    format: GgufBlockFormat,
    retained: Experiment,
    word_pack: CudaFunction,
    word_scalar: CudaFunction,
    word_tiled: CudaFunction,
}

impl LayoutExperiment {
    fn load(context: &Arc<CudaContext>, format: GgufBlockFormat) -> Self {
        let retained = Experiment::load(context); // Includes the DP4A capability check.
        let module = context
            .load_module(Ptx::from_src(crate::ptx::VNEXT_GGUF.to_owned()))
            .unwrap();
        let load = |name| module.load_function(name).unwrap();
        let (scalar, tiled) = match format {
            GgufBlockFormat::Iq4Xs => (
                "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                "vnext_gguf_iq4xs_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            ),
            GgufBlockFormat::Q4K => (
                "vnext_gguf_q4k_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                "vnext_gguf_q4k_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            ),
            GgufBlockFormat::Q5K => (
                "vnext_gguf_q5k_q8_f32scale_dp4a_group32_word_major_f16_prototype",
                "vnext_gguf_q5k_q8_f32scale_dp4a_group32_word_major_tiled_f16_prototype",
            ),
            other => panic!("unsupported layout format {other:?}"),
        };
        Self {
            format,
            retained,
            word_pack: load("vnext_gguf_q8_f32scale_pack_word_major_f16_prototype"),
            word_scalar: load(scalar),
            word_tiled: load(tiled),
        }
    }

    fn pack(&self, case: &mut Fixture, stream: &Arc<CudaStream>, layout: QwordLayout) {
        case.pack_kernel(
            stream,
            match layout {
                QwordLayout::GroupMajor => &self.retained.quantize,
                QwordLayout::WordMajor => &self.word_pack,
            },
        );
    }

    fn prepare(&self, case: &mut Fixture, stream: &Arc<CudaStream>, route: LayoutRoute) {
        case.reset(stream);
        if !route.includes_pack() {
            self.pack(case, stream, route.layout());
        }
        stream.synchronize().unwrap();
    }

    fn multiply(
        &self,
        case: &mut Fixture,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        route: LayoutRoute,
        tile: u32,
    ) {
        assert_eq!(
            case.format, self.format,
            "layout kernel/weight format mismatch"
        );
        if matches!(route.layout(), QwordLayout::GroupMajor) {
            case.multiply(
                stream,
                kernels,
                &self.retained,
                route.numerical_route(),
                tile,
            );
            return;
        }
        let function = match tile {
            1 => &self.word_scalar,
            8 => &self.word_tiled,
            _ => panic!("unsupported layout row tile"),
        };
        let groups = case.rows * case.inputs / 32;
        let scales = case.scales_gpu.slice(SCALE_PREFIX..SCALE_PREFIX + groups);
        let words = case.words_gpu.slice(WORD_PREFIX..WORD_PREFIX + groups * 8);
        let weights = case
            .weights_gpu
            .slice(WEIGHT_PREFIX..case.weights.len() - 7);
        let mut output = case
            .output_gpu
            .slice_mut(OUTPUT_PREFIX..OUTPUT_PREFIX + case.rows * case.stride);
        let params = [
            case.rows as u32,
            case.inputs as u32,
            case.outputs as u32,
            case.stride as u32,
            COLUMN_OFFSET as u32,
        ];
        let mut launch = stream.launch_builder(function);
        launch
            .arg(&scales)
            .arg(&words)
            .arg(&weights)
            .arg(&mut output);
        for value in &params {
            launch.arg(value);
        }
        // SAFETY: same G32 ABI and extents; only the qword permutation differs.
        // Weights remain byte-safe at odd offset5, output columns/rows have tails.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (
                    (case.outputs as u32).div_ceil(4),
                    (case.rows as u32).div_ceil(tile),
                    1,
                ),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .unwrap();
    }

    fn enqueue(
        &self,
        case: &mut Fixture,
        stream: &Arc<CudaStream>,
        kernels: &CudaNativeBlockKernels,
        route: LayoutRoute,
        tile: u32,
        iterations: u32,
    ) {
        for _ in 0..iterations {
            if route.includes_pack() {
                self.pack(case, stream, route.layout());
            }
            self.multiply(case, stream, kernels, route, tile);
        }
    }

    fn validate(
        &self,
        case: &Fixture,
        stream: &Arc<CudaStream>,
        route: LayoutRoute,
    ) -> (Vec<u16>, serde_json::Value) {
        case.validate_pack_layout(stream, route.layout());
        case.validate_output(stream, route.numerical_route())
    }
}

fn output_bits(case: &Fixture, stream: &Arc<CudaStream>) -> Vec<u16> {
    stream
        .clone_dtoh(&case.output_gpu)
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

// Canonical logical order allows comparing layouts without changing device data.
fn row_bits(
    case: &Fixture,
    stream: &Arc<CudaStream>,
    layout: QwordLayout,
    row: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u16>) {
    let groups = case.inputs / 32;
    let scales = stream.clone_dtoh(&case.scales_gpu).unwrap();
    let words = stream.clone_dtoh(&case.words_gpu).unwrap();
    let output = output_bits(case, stream);
    let canonical_words = (0..groups)
        .flat_map(|group| {
            let words = &words;
            (0..8).map(move |word| words[WORD_PREFIX + layout.physical(row, group, word, groups)])
        })
        .collect();
    (
        scales[SCALE_PREFIX + row * groups..SCALE_PREFIX + (row + 1) * groups]
            .iter()
            .map(|x| x.to_bits())
            .collect(),
        canonical_words,
        output[OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET
            ..OUTPUT_PREFIX + row * case.stride + COLUMN_OFFSET + case.outputs]
            .to_vec(),
    )
}
