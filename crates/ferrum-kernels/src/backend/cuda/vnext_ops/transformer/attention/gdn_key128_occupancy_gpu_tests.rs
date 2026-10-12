//! Actual old/new CUDA entry oracle. This raw fixture owns every live device
//! allocation; it does not mint Core admission or cross-wave permissions.
use super::{DELTA_KEY128_INTERLEAVED_FUNCTION, DELTA_TILED_INTERLEAVED_FUNCTION};
use crate::backend::cuda::vnext_ops::transformer::test_support::Guarded;
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const DIM: usize = 128;
const STATE_ELEMENTS: usize = VALUE_HEADS * DIM * DIM;
const OUTPUT_ELEMENTS: usize = VALUE_HEADS * DIM;
const POISON: f32 = 193.0;

fn signal(elements: usize, salt: usize, denominator: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| (((index * 17 + salt * 31) % 101) as f32 - 50.0) / denominator)
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

struct StateArm {
    owners: Vec<Guarded<f32>>,
    unused_even_slot: Guarded<f32>,
}

impl StateArm {
    fn new(stream: &Arc<CudaStream>, rows: usize) -> Self {
        Self {
            owners: (0..rows)
                .map(|owner| {
                    Guarded::new(stream, &signal(STATE_ELEMENTS, owner + 7, 4096.0), POISON)
                })
                .collect(),
            // A real, fully bounded decoy makes an accidental even-slot write
            // detectable without supplying an unsafe or freed pointer.
            unused_even_slot: Guarded::new(stream, &vec![POISON; STATE_ELEMENTS], POISON),
        }
    }

    fn bindings(&self, stream: &Arc<CudaStream>, order: &[usize]) -> Guarded<u64> {
        let even = self.unused_even_slot.pointer(stream);
        let pointers = order
            .iter()
            .flat_map(|&owner| [even, self.owners[owner].pointer(stream)])
            .collect::<Vec<_>>();
        Guarded::new(stream, &pointers, 0xfedc_ba98_7654_3210)
    }
}

struct Inputs {
    query: Guarded<f32>,
    key: Guarded<f32>,
    value: Guarded<f32>,
    g: Guarded<f32>,
    beta: Guarded<f32>,
    cu_seqlens: Guarded<u32>,
    tokens: usize,
}

impl Inputs {
    fn new(stream: &Arc<CudaStream>, lengths: &[usize], step: usize) -> Self {
        let mut cumulative = vec![0u32];
        for &length in lengths {
            cumulative.push(cumulative.last().unwrap() + length as u32);
        }
        let tokens = *cumulative.last().unwrap() as usize;
        Self {
            query: Guarded::new(
                stream,
                &signal(tokens * KEY_HEADS * DIM, 11 + step, 128.0),
                POISON,
            ),
            key: Guarded::new(
                stream,
                &signal(tokens * KEY_HEADS * DIM, 23 + step, 512.0),
                POISON,
            ),
            value: Guarded::new(
                stream,
                &signal(tokens * OUTPUT_ELEMENTS, 37 + step, 64.0),
                POISON,
            ),
            g: Guarded::new(stream, &vec![-0.125; tokens * VALUE_HEADS], POISON),
            beta: Guarded::new(stream, &vec![0.375; tokens * VALUE_HEADS], POISON),
            cu_seqlens: Guarded::new(stream, &cumulative, 0xfedc_ba98),
            tokens,
        }
    }

    fn assert_unchanged(&self, stream: &Arc<CudaStream>) {
        self.query.assert_unchanged(stream);
        self.key.assert_unchanged(stream);
        self.value.assert_unchanged(stream);
        self.g.assert_unchanged(stream);
        self.beta.assert_unchanged(stream);
        self.cu_seqlens.assert_unchanged(stream);
    }
}

fn launch(
    stream: &Arc<CudaStream>,
    function: &CudaFunction,
    inputs: &Inputs,
    bindings: &Guarded<u64>,
    output: &Guarded<f32>,
    rows: usize,
) {
    let pointers = [
        inputs.query.pointer(stream),
        inputs.key.pointer(stream),
        inputs.value.pointer(stream),
        inputs.g.pointer(stream),
        inputs.beta.pointer(stream),
        bindings.pointer(stream),
        inputs.cu_seqlens.pointer(stream),
        output.pointer(stream),
    ];
    let dimensions = [
        rows as i32,
        inputs.tokens as i32,
        KEY_HEADS as i32,
        VALUE_HEADS as i32,
        DIM as i32,
        DIM as i32,
    ];
    let scale = (DIM as f32).sqrt().recip();
    let mut builder = stream.launch_builder(function);
    for pointer in &pointers {
        builder.arg(pointer);
    }
    for dimension in &dimensions {
        builder.arg(dimension);
    }
    builder.arg(&scale);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: ((DIM as u32).div_ceil(16), VALUE_HEADS as u32, rows as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .unwrap();
}

#[test]
#[ignore = "requires an exclusive actual CUDA device and current old/new GDN PTX"]
fn gdn_key128_dedicated_matches_original_output_and_full_state() {
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();
    let module = context
        .load_module(Ptx::from_src(crate::ptx::GATED_DELTA_RULE))
        .unwrap();
    let old = module
        .load_function(DELTA_TILED_INTERLEAVED_FUNCTION)
        .unwrap();
    let new = module
        .load_function(DELTA_KEY128_INTERLEAVED_FUNCTION)
        .unwrap();
    let old_shared = old.shared_size_bytes().unwrap();
    let new_shared = new.shared_size_bytes().unwrap();
    assert_eq!(
        new_shared, 0,
        "dedicated entry must not reserve generic fallback scratch"
    );
    eprintln!("GDN actual driver resource attributes old={{static_shared_bytes:{old_shared},registers:{},local_bytes:{}}} new={{static_shared_bytes:{new_shared},registers:{},local_bytes:{}}}",
        old.num_regs().unwrap(), old.local_size_bytes().unwrap(),
        new.num_regs().unwrap(), new.local_size_bytes().unwrap());

    for rows in [8usize, 32] {
        let mut arms = [StateArm::new(&stream, rows), StateArm::new(&stream, rows)];
        let mut physical = arms
            .iter()
            .flat_map(|arm| arm.owners.iter())
            .map(|owner| {
                let start = owner.pointer(&stream);
                (start, start + (STATE_ELEMENTS * 4) as u64)
            })
            .collect::<Vec<_>>();
        physical.sort_unstable();
        assert!(
            physical.windows(2).all(|pair| pair[0].1 <= pair[1].0),
            "oracle states must be independent real allocations"
        );
        let mut previous = arms[0]
            .owners
            .iter()
            .map(|owner| bits(&owner.read(&stream)))
            .collect::<Vec<_>>();
        let mut retired = Vec::new();
        for step in 0..3 {
            if step == 2 {
                let initial = signal(STATE_ELEMENTS, rows + 91, 4096.0);
                previous[1] = bits(&initial);
                for arm in &mut arms {
                    let old_owner = std::mem::replace(
                        &mut arm.owners[1],
                        Guarded::new(&stream, &initial, POISON),
                    );
                    let snapshot = bits(&old_owner.read(&stream));
                    retired.push((old_owner, snapshot));
                }
            }
            let order = (0..rows)
                .map(|row| (row * 5 + step * 3) % rows)
                .collect::<Vec<_>>();
            let lengths = order
                .iter()
                .enumerate()
                .map(|(row, &owner)| {
                    if step == 1 && row + 1 == rows {
                        0
                    } else {
                        1 + (owner + step) % 3
                    }
                })
                .collect::<Vec<_>>();
            let inputs = Inputs::new(&stream, &lengths, step);
            let bindings = arms.each_ref().map(|arm| arm.bindings(&stream, &order));
            // Meaningful non-tile padding is outside the actual output span;
            // the typed value dimension stays the supported128, not a fake130.
            let output_length = inputs.tokens * OUTPUT_ELEMENTS;
            let outputs = [
                Guarded::new(&stream, &vec![POISON; output_length + 13], POISON),
                Guarded::new(&stream, &vec![POISON; output_length + 13], POISON),
            ];
            launch(&stream, &old, &inputs, &bindings[0], &outputs[0], rows);
            launch(&stream, &new, &inputs, &bindings[1], &outputs[1], rows);
            stream.synchronize().unwrap();
            let result = outputs.each_ref().map(|output| output.read(&stream));
            assert!(
                result[0][..output_length]
                    .iter()
                    .all(|value| value.is_finite() && *value != POISON),
                "every active output element must be written"
            );
            assert_eq!(
                bits(&result[0]),
                bits(&result[1]),
                "entire GDN output changed at M{rows}/step{step}"
            );
            for values in &result {
                assert_eq!(&values[output_length..], &[POISON; 13]);
            }
            for owner in 0..rows {
                let before = &previous[owner];
                let state = arms
                    .each_ref()
                    .map(|arm| bits(&arm.owners[owner].read(&stream)));
                assert_eq!(
                    state[0], state[1],
                    "entire recurrent state changed for owner{owner}/M{rows}/step{step}"
                );
                let row = order
                    .iter()
                    .position(|candidate| *candidate == owner)
                    .unwrap();
                if lengths[row] == 0 {
                    assert_eq!(
                        &state[0], before,
                        "empty row must preserve its current state"
                    );
                } else {
                    assert_ne!(
                        &state[0], before,
                        "nonempty history must exercise recurrent updates"
                    );
                }
                previous[owner] = state[0].clone();
            }
            inputs.assert_unchanged(&stream);
            for binding in &bindings {
                binding.assert_unchanged(&stream);
            }
            for arm in &arms {
                arm.unused_even_slot.assert_unchanged(&stream);
            }
            for (owner, snapshot) in &retired {
                assert_eq!(
                    &bits(&owner.read(&stream)),
                    snapshot,
                    "replaced owner must no longer be addressed"
                );
            }
        }
        eprintln!("GDN old/new complete output and state bits equal: actual_rows={rows}, key_heads={KEY_HEADS}, value_heads={VALUE_HEADS}, key_dim={DIM}, value_dim={DIM}; varlen history, reordered bindings, fresh owner, empty row, guards and output padding checked");
    }
}
