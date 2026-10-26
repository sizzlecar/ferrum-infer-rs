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
    let words = [
        0, 0x8000, 0x0001, 0x007f, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffd5, 0x3f80, 0xc000,
    ];
    let source = |n: usize, seed| {
        (0..n)
            .map(|i| bf16::from_bits(words[(i + seed) % words.len()]).to_f32())
            .collect::<Vec<_>>()
    };
    let check = |reference: &[f32], actual: &[f32]| {
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        )
    };
    // All 32 native run slots, repeated/consecutive rows and per-batch offset,
    // interior column window, non-aligned tails, NaN payloads and signed zero.
    let values = source(64 * 19, 0);
    let mut attrs = BTreeMap::from([
        (id("batch"), SemanticValue::Unsigned(2)),
        (id("column_start"), SemanticValue::Unsigned(1)),
        (id("column_count"), SemanticValue::Unsigned(17)),
    ]);
    for (name, list) in [
        ("run_lengths", vec![2; 32]),
        ("source_starts", (0..32).map(|i| i as i64).collect()),
        ("source_batch_strides", vec![16; 32]),
        (
            "source_row_strides",
            (0..32).map(|i| (i % 2) as i64).collect(),
        ),
    ] {
        attrs.insert(id(name), SemanticValue::Integers(list));
    }
    let mut fixture = Fixture::new(
        MetalTensorOperation::RowGather,
        vec![spec(vec![64, 19], ElementType::Bf16)],
        attrs,
    );
    fixture.output = Some(spec(vec![128, 17], ElementType::Bf16));
    let mut reference = Vec::new();
    for batch in 0..2 {
        for run in 0..32 {
            for step in 0..2 {
                let row = run + batch * 16 + step * (run % 2);
                reference.extend_from_slice(&values[row * 19 + 1..row * 19 + 18]);
            }
        }
    }
    run_case(
        runtime,
        registry,
        catalog,
        &policy,
        host,
        &fixture,
        &[values],
        |actual| check(&reference, actual),
    );
    let left = source(6 * 17, 3);
    let right = source(4 * 17, 7);
    let mut fixture = Fixture::new(
        MetalTensorOperation::BatchedRowConcat,
        vec![
            spec(vec![6, 17], ElementType::Bf16),
            spec(vec![4, 17], ElementType::Bf16),
        ],
        BTreeMap::from([(id("batch"), SemanticValue::Unsigned(2))]),
    );
    fixture.output = Some(spec(vec![10, 17], ElementType::Bf16));
    let mut reference = Vec::new();
    for batch in 0..2 {
        reference.extend_from_slice(&left[batch * 51..(batch + 1) * 51]);
        reference.extend_from_slice(&right[batch * 34..(batch + 1) * 34]);
    }
    run_case(
        runtime,
        registry,
        catalog,
        &policy,
        host,
        &fixture,
        &[left, right],
        |actual| check(&reference, actual),
    );
    eprintln!("Metal row transport: exact BF16 bits including NaN payloads, signed zeros, 32 runs, column windows, per-batch concat and retirement passed");
}
