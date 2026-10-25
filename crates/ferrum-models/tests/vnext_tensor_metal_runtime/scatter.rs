use super::*;
use ferrum_execution::component::TensorComponentPolicy;
use transforms::{fixture::Fixture, run_case, spec};

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    host: &HostMemoryAccount,
) {
    let policy = TensorComponentPolicy::new(
        CAPACITY,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    for runs in [
        vec![(0_u64, 100_u64)],
        (0..32).map(|i| (3 * i + 1, 1 + i % 2)).collect(),
    ] {
        let update_rows: u64 = runs.iter().map(|(_, n)| n).sum();
        let fixture = Fixture::new(
            MetalTensorOperation::RowScatterAdd,
            vec![
                spec(vec![100, 17], ElementType::Bf16),
                spec(vec![update_rows, 17], ElementType::Bf16),
            ],
            BTreeMap::from([
                (
                    id("destination_starts"),
                    SemanticValue::Integers(runs.iter().map(|(s, _)| *s as i64).collect()),
                ),
                (
                    id("run_lengths"),
                    SemanticValue::Integers(runs.iter().map(|(_, n)| *n as i64).collect()),
                ),
            ]),
        );
        // Ties, cancellation, signs, overflow, nonfinite values; untouched rows
        // additionally retain BF16 subnormals and NaN payloads bit for bit.
        let words = [
            0, 0x8000, 0x3f80, 0x3f81, 0xbf80, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffd5,
        ];
        let mut input: Vec<_> = (0..1700)
            .map(|i| bf16::from_bits(words[i % words.len()]).to_f32())
            .collect();
        for row in 0..100 {
            if !runs
                .iter()
                .any(|&(start, n)| (start..start + n).contains(&(row as u64)))
            {
                input[row * 17] = bf16::from_bits(1).to_f32();
            }
        }
        let updates: Vec<_> = (0..update_rows as usize * 17)
            .map(|i| {
                [
                    0.0,
                    -0.0,
                    1.0 / 256.0,
                    -1.0,
                    bf16::MAX.to_f32(),
                    f32::INFINITY,
                ][i % 6]
            })
            .collect();
        let mut reference = input.clone();
        let mut touched = vec![false; input.len()];
        for (row, update) in runs
            .iter()
            .flat_map(|&(start, n)| start..start + n)
            .map(|r| r as usize)
            .zip(updates.chunks_exact(17))
        {
            for (destination, delta) in reference[row * 17..(row + 1) * 17].iter_mut().zip(update) {
                *destination = rounds(*destination + delta);
            }
            touched[row * 17..(row + 1) * 17].fill(true);
        }
        run_case(
            runtime,
            registry,
            catalog,
            &policy,
            host,
            &fixture,
            &[input, updates],
            |actual| {
                assert_eq!(actual.len(), reference.len());
                for (i, (&actual, &expected)) in actual.iter().zip(&reference).enumerate() {
                    if touched[i] && expected.is_nan() {
                        assert!(actual.is_nan(), "scatter element {i}");
                    } else {
                        assert_eq!(actual.to_bits(), expected.to_bits(), "scatter element {i}");
                    }
                }
            },
        );
    }
    eprintln!("Metal scatter: full and sparse 32-run updates, untouched payloads, rounding, exceptional values and retirement passed");
}
