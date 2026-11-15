use super::super::cuda_support::{self, id};
use super::*;

#[test]
#[ignore = "requires locked SM120 NVFP4 and tensor modules; independent exact subset, not full decoder parity"]
fn original_video_decoder_complete_block_runs_managed_nvfp4_attention_and_staged_ffn() {
    qualify_sequences(vec![2, 3], false);
}

#[test]
#[ignore = "requires locked SM120 NVFP4 and explicit cuDNN; real independent native batches, ragged groups and capacity tail"]
fn cudnn_video_decoder_block_keeps_grouped_and_ragged_sequences_isolated() {
    let mut lengths = vec![2; 17];
    lengths.extend([3, 3, 2]);
    qualify_sequences(lengths, true);
}

fn qualify_sequences(lengths: Vec<u64>, use_cudnn: bool) {
    let (_dir, package) = fixture().open(limits());
    let account = cuda_support::host();
    let signs: Vec<f32> = lengths
        .iter()
        .enumerate()
        .flat_map(|(sequence, &length)| {
            std::iter::repeat_n(if sequence % 2 == 0 { 1. } else { -1. }, length as usize)
        })
        .collect();
    let rows = signs.len();
    let (config, source) = FastH3VideoDecoderBlocksSource::prepare(
        Arc::new(package.unwrap()),
        0..1,
        lengths,
        &account,
        block_limits(),
    )
    .unwrap();
    let prepared = TypedFamilyRegistration::new(F::new(block_limits()).unwrap())
        .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(F::PROFILE_ID))
        .unwrap();
    let mut inputs = Vec::new();
    for (name, width, dtype) in [
        (F::INPUT_ID, 128, ElementType::F32),
        (F::COSINE_ID, 48, ElementType::Bf16),
        (F::SINE_ID, 48, ElementType::Bf16),
    ] {
        let mut bytes = account
            .allocate_zeroed(
                HostMemoryClass::InputPreprocess,
                rows * width * if dtype == ElementType::F32 { 4 } else { 2 },
            )
            .unwrap();
        if dtype == ElementType::F32 {
            for (i, out) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
                out.copy_from_slice(&signs[i / 128].to_le_bytes());
            }
        } else if name == F::COSINE_ID {
            for out in bytes.as_mut().chunks_exact_mut(2) {
                out.copy_from_slice(&0x3f80u16.to_le_bytes());
            }
        }
        inputs.push(
            ManagedHostTensor::new(
                id(name),
                ResolvedTensorSpec::new(
                    vec![rows as u64, width as u64],
                    dtype,
                    ResolvedTensorLayout::Contiguous,
                )
                .unwrap(),
                bytes.freeze(),
            )
            .unwrap(),
        );
    }
    let execute = if use_cudnn {
        cuda_support::execute_twice_with_cudnn
    } else {
        cuda_support::execute_twice
    };
    let retained = execute(
        prepared,
        config.compile_options(block_limits()).unwrap(),
        Arc::new(source),
        inputs,
        &account,
        |outputs: &[ferrum_execution::TensorExecutionOutput]| {
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].value_id(), &config.output_id().unwrap());
            // Independent analytic oracle: nonzero diagonal GEMMs; per-sequence
            // constant V makes attention independent of score approximation.
            for (i, word) in outputs[0].bytes().bytes().chunks_exact(4).enumerate() {
                assert_eq!(
                    u32::from_le_bytes(word.try_into().unwrap()),
                    (if signs[i / 128] > 0. {
                        14.625f32
                    } else {
                        -14.375f32
                    })
                    .to_bits(),
                    "row {} column {}",
                    i / 128,
                    i % 128
                );
            }
        },
    );
    assert!(account.usage().unwrap().committed_bytes() > 0);
    drop(retained);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    eprintln!("Original video decoder block: D64/rot48, {rows} rows with independent sequences, cuDNN={use_cudnn}, six nonzero NVFP4 projections, staged BF16 QK/FFN, F32 separate residuals, independent exact subset");
}
