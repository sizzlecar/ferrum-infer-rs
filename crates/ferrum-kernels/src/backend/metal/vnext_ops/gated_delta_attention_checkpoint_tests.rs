//! Complete conv/delta state continuation of the real recurrent core kernels.
//! Projected inputs are supplied directly: this is not evidence for the outer
//! quantized projections, residual path, or public checkpoint admission.

use super::*;
use std::{mem::size_of, ops::Range};

#[path = "checkpoint_copy_test_support.rs"]
mod native;

const PREFIX: usize = 5;
const TOTAL: usize = 9;
const CONV_BYTES: u64 = (QKV_FEATURES * CONV_STATE_WIDTH * size_of::<f16>()) as u64;
const DELTA_BYTES: u64 = (VALUE_HEADS * VALUE_DIM * KEY_DIM * size_of::<f32>()) as u64;

#[test]
fn private_native_checkpoint_preserves_gated_delta_state_and_continuation_bits() {
    metal::objc::rc::autoreleasepool(|| {
        let device = Device::system_default().expect("checkpoint continuation requires Metal");
        let queue = device.new_command_queue();
        let pipelines = MetalGatedDeltaPipelines::new(&device).unwrap();
        let inputs = Inputs {
            mixed_qkv: half_values(TOTAL * QKV_FEATURES, 0.017, 0.21),
            a: half_values(TOTAL * VALUE_HEADS, 0.071, 0.18),
            b: half_values(TOTAL * VALUE_HEADS, 0.053, 0.24),
            z: half_values(TOTAL * VALUE_FEATURES, 0.029, 0.31),
        };
        let conv = shared_buffer(
            &device,
            &half_values(QKV_FEATURES * CONV_KERNEL, 0.011, 0.16),
        );
        let negative_rates = (0..VALUE_HEADS)
            .map(|index| -(-1.7 + index as f32 * 0.07).exp())
            .collect::<Vec<_>>();
        let decay = shared_buffer(&device, &negative_rates);
        let dt_bias = shared_buffer(
            &device,
            &(0..VALUE_HEADS)
                .map(|index| -0.25 + index as f32 * 0.04)
                .collect::<Vec<_>>(),
        );
        let norm = shared_buffer(
            &device,
            &(0..VALUE_DIM)
                .map(|index| 0.82 + index as f32 * 0.013)
                .collect::<Vec<_>>(),
        );
        let weights = StaticWeights {
            conv: &conv,
            decay_parameter: &decay,
            dt_bias: &dt_bias,
            norm: &norm,
        };
        // This is the native GGUF recurrent ABI, without reinterpreting its
        // negative decay rates or interleaved value-head mapping.
        let semantics = TestSemantics {
            decay_parameterization: GatedDeltaDecayParameterization::NegativeRate,
            value_head_mapping: GatedDeltaValueHeadMapping::InterleavedByKeyHead,
        };
        // Identical single-participant spans and SIMD-recurrent dispatch in
        // every pair. The first span is shorter than the conv history width.
        let prefix = [0..2, 2..PREFIX];
        let suffix = [PREFIX..8, 8..TOTAL];
        eprintln!("gated-delta native continuation: prefix={prefix:?}, suffix={suffix:?}, participants=1, projected-input ABI, key_heads=16 value_heads=32 key_dim=128 value_dim=128, conv=8192x3/F16 delta=32x128x128/F32, negative-rate/interleaved, SIMD recurrent");
        let source = State::zeroed(&device, &queue);
        let cold = State::zeroed(&device, &queue);
        let restored = State::filled(&device, &queue, 0xCD);
        let prefix_output = inputs.run(
            &device, &queue, &pipelines, &weights, &source, semantics, &prefix,
        );
        let cold_prefix_output = inputs.run(
            &device, &queue, &pipelines, &weights, &cold, semantics, &prefix,
        );
        native::assert_output_bits("cold prefix", &cold_prefix_output, &prefix_output);
        let saved = source.read(&device, &queue);
        native::assert_bits(
            "cold prefix conv and delta",
            &cold.read(&device, &queue),
            &saved,
        );
        let checkpoint = native::private_filled(&device, &queue, CONV_BYTES + DELTA_BYTES, 0xAB);
        native::copy(&queue, &source.conv, 0, &checkpoint, 0, CONV_BYTES);
        native::copy(
            &queue,
            &source.delta,
            0,
            &checkpoint,
            CONV_BYTES,
            DELTA_BYTES,
        );
        native::assert_bits(
            "captured complete conv and delta",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved,
        );
        let source_output = inputs.run(
            &device, &queue, &pipelines, &weights, &source, semantics, &suffix,
        );
        assert_ne!(
            source.read(&device, &queue),
            saved,
            "suffix must actually update recurrent state"
        );
        native::assert_bits(
            "checkpoint after source continuation",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved,
        );
        native::copy(&queue, &checkpoint, 0, &restored.conv, 0, CONV_BYTES);
        native::copy(
            &queue,
            &checkpoint,
            CONV_BYTES,
            &restored.delta,
            0,
            DELTA_BYTES,
        );
        native::assert_bits(
            "restored complete conv and delta",
            &restored.read(&device, &queue),
            &saved,
        );
        let cold_output = inputs.run(
            &device, &queue, &pipelines, &weights, &cold, semantics, &suffix,
        );
        let restored_output = inputs.run(
            &device, &queue, &pipelines, &weights, &restored, semantics, &suffix,
        );
        native::assert_output_bits("cold suffix", &cold_output, &source_output);
        native::assert_output_bits("restored suffix", &restored_output, &source_output);
        let final_source = source.read(&device, &queue);
        native::assert_bits(
            "cold final conv and delta",
            &cold.read(&device, &queue),
            &final_source,
        );
        native::assert_bits(
            "restored final conv and delta",
            &restored.read(&device, &queue),
            &final_source,
        );
        native::assert_bits(
            "checkpoint after all branches",
            &native::read_bytes(&device, &queue, &checkpoint),
            &saved,
        );
    });
}

struct State {
    conv: metal::Buffer,
    delta: metal::Buffer,
}

impl State {
    fn zeroed(device: &Device, queue: &CommandQueueRef) -> Self {
        Self::filled(device, queue, 0)
    }

    fn filled(device: &Device, queue: &CommandQueueRef, fill: u8) -> Self {
        Self {
            conv: native::private_filled(device, queue, CONV_BYTES, fill),
            delta: native::private_filled(device, queue, DELTA_BYTES, fill),
        }
    }

    fn read(&self, device: &Device, queue: &CommandQueueRef) -> Vec<u8> {
        let mut bytes = native::read_bytes(device, queue, &self.conv);
        bytes.extend(native::read_bytes(device, queue, &self.delta));
        bytes
    }
}

struct Inputs {
    mixed_qkv: Vec<f16>,
    a: Vec<f16>,
    b: Vec<f16>,
    z: Vec<f16>,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        device: &Device,
        queue: &CommandQueueRef,
        pipelines: &MetalGatedDeltaPipelines,
        weights: &StaticWeights<'_>,
        state: &State,
        semantics: TestSemantics,
        spans: &[Range<usize>],
    ) -> Vec<u16> {
        spans
            .iter()
            .flat_map(|span| {
                run_segment_bits(
                    device,
                    queue,
                    pipelines,
                    segment(&self.mixed_qkv, &self.a, &self.b, &self.z, span.clone()),
                    weights,
                    &state.conv,
                    &state.delta,
                    semantics,
                )
            })
            .collect()
    }
}
