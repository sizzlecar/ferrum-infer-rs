use super::*;

fn launch(rows: u64, format: LinearPhysicalFormat, activation: ElementType) -> LinearLaunch {
    linear_launch_typed(
        PreparedLinearPart {
            region: 1,
            format,
            output_offset: 3,
            out_features: 1024,
            transform: None,
        },
        0,
        2,
        rows,
        2048,
        1031,
        16,
        32,
        activation,
    )
    .unwrap()
}

#[test]
fn metal_linear_prepared_classes_match_full_class_and_numeric_evidence() {
    for format in [
        LinearPhysicalFormat::DenseF16,
        LinearPhysicalFormat::Q4K,
        LinearPhysicalFormat::Q5K,
        LinearPhysicalFormat::Q6K,
        LinearPhysicalFormat::Q8_0,
    ] {
        for activation in [ElementType::F16, ElementType::F32] {
            let prepared = PreparedLinearClasses::new(launch(1, format, activation)).unwrap();
            for rows in [1, 4, 8, 33, 768] {
                let launch = launch(rows, format, activation);
                let mut original = SelectedCommandCostBuilderV1::new_with_algorithm_work(rows);
                let mut compiled = SelectedCommandCostBuilderV1::new_with_algorithm_work(rows);
                for &key in CLASSES {
                    let mut geometry = key.geometry();
                    geometry.groups = [rows as u32, 17, 1];
                    geometry.padded_outputs = rows * 1088;
                    let old = class(key.entry, launch, geometry, key.staged).unwrap();
                    let new = prepared
                        .get(key.entry, launch, geometry, key.staged)
                        .unwrap();
                    assert_eq!(new, old, "full ABI and selected block: {}", key.entry);
                    let work = KernelNumericWorkV1 {
                        logical_units: rows * 1024,
                        padded_units: geometry.padded_outputs,
                        inner_units_per_logical_unit: 2048,
                        grid: geometry.groups,
                        scratch_bytes: rows * 4096,
                        staged_weight_bytes: if key.staged { 2048 * 1024 * 2 } else { 0 },
                    };
                    original.kernel(old, work).unwrap();
                    compiled.kernel(new, work).unwrap();
                }
                let old = original.finish().unwrap();
                let new = compiled.finish().unwrap();
                assert_eq!(new, old);
                assert_eq!(new.algorithm_work(), old.algorithm_work());
                assert_eq!(
                    new.independent_attention_family_v2(),
                    old.independent_attention_family_v2()
                );
            }
        }
    }
}

#[test]
fn metal_linear_prepared_classes_reject_abi_or_selected_block_drift() {
    let current = launch(8, LinearPhysicalFormat::Q4K, ElementType::F16);
    let prepared = PreparedLinearClasses::new(current).unwrap();
    let key = CLASSES[0];
    let expected = prepared
        .get(key.entry, current, key.geometry(), key.staged)
        .unwrap();
    for field in 0..6 {
        let mut changed = current;
        match field {
            0 => changed.params.in_features += 1,
            1 => changed.params.out_features += 1,
            2 => changed.params.output_stride += 1,
            3 => changed.params.output_column_offset += 1,
            4 => changed.activation_type = ElementType::F32,
            5 => changed.format = LinearPhysicalFormat::Q6K,
            _ => unreachable!(),
        }
        assert!(prepared
            .get(key.entry, changed, key.geometry(), key.staged)
            .is_none());
    }
    let mut dynamic = current;
    dynamic.params.rows = 33;
    dynamic.input_offset_bytes += 64;
    dynamic.output_offset_bytes += 128;
    let mut grid = key.geometry();
    grid.groups = [33, 257, 1];
    grid.padded_outputs = 33 * 1028;
    assert_eq!(
        prepared.get(key.entry, dynamic, grid, key.staged),
        Some(expected)
    );
    assert!(prepared
        .get("unmapped.actual.pipeline", current, grid, key.staged)
        .is_none());
    grid.threads[0] += 1;
    assert!(prepared.get(key.entry, current, grid, key.staged).is_none());
    grid = key.geometry();
    grid.threadgroup_memory = Some(1);
    assert!(prepared.get(key.entry, current, grid, key.staged).is_none());
    assert!(prepared
        .get(key.entry, current, key.geometry(), !key.staged)
        .is_none());
}

#[test]
#[ignore = "CPU class preparation and lookup diagnostic; no performance threshold"]
fn metal_linear_prepared_classes_cpu_timing() {
    use std::{hint::black_box, time::Instant};
    let launch = launch(8, LinearPhysicalFormat::Q4K, ElementType::F16);
    // Shader/source hashing is already OnceLock on both paths; exclude its
    // one-time warmup rather than attributing it to repeated class work.
    let prepared = PreparedLinearClasses::new(launch).unwrap();
    for &key in CLASSES {
        assert_eq!(
            prepared.get(key.entry, launch, key.geometry(), key.staged),
            class(key.entry, launch, key.geometry(), key.staged)
        );
    }
    let preparation_iterations = 100_u32;
    let start = Instant::now();
    for _ in 0..preparation_iterations {
        black_box(PreparedLinearClasses::new(black_box(launch)).unwrap());
    }
    let preparation_ns = start.elapsed().as_nanos() / u128::from(preparation_iterations);
    let iterations = 2000_u32;
    let start = Instant::now();
    for _ in 0..iterations {
        for &key in CLASSES {
            let key = black_box(key);
            black_box(class(key.entry, black_box(launch), key.geometry(), key.staged).unwrap());
        }
    }
    let original_per_class_ns =
        start.elapsed().as_nanos() / u128::from(iterations) / CLASSES.len() as u128;
    let start = Instant::now();
    for _ in 0..iterations {
        for &key in CLASSES {
            let key = black_box(key);
            black_box(
                prepared
                    .get(key.entry, black_box(launch), key.geometry(), key.staged)
                    .unwrap(),
            );
        }
    }
    let prepared_per_class_ns =
        start.elapsed().as_nanos() / u128::from(iterations) / CLASSES.len() as u128;
    let retained_bytes = std::mem::size_of_val(&prepared)
        + prepared.entries.capacity() * std::mem::size_of::<(ClassKey, SelectedAlgorithmClassV1)>();
    eprintln!("metal_linear_prepared_classes classes={} preparation_ns={preparation_ns} retained_payload_bytes={retained_bytes} fresh_per_class_ns={original_per_class_ns} prepared_per_class_ns={prepared_per_class_ns}; excludes unchanged actual PSO selection, dynamic numeric builder and resource projection", CLASSES.len());
}
