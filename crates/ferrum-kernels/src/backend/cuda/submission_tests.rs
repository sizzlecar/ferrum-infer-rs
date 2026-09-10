//! Exercise the legacy Backend stream, independently of the vNext runtime.
use super::{Backend, CudaBackend};
use half::f16;

#[test]
#[ignore = "requires an actual CUDA device and the legacy native kernels"]
fn legacy_stream_preserves_upload_compute_copy_order_and_context_reuse() {
    // Non-tile dimensions exercise GEMM edges and the elementwise launch tail.
    let (rows, columns, inner) = (3, 17, 19);
    let count = rows * columns;
    let guard = 7;
    let sentinel = -127.0;
    let input: Vec<_> = (0..rows * inner)
        .map(|index| ((index % 7) as f32 - 3.0) / 8.0)
        .collect();
    let weights: Vec<_> = (0..columns * inner)
        .map(|index| ((index % 9) as f32 - 4.0) / 16.0)
        .collect();
    let expected: Vec<_> = (0..count)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            let sum: f64 = (0..inner)
                .map(|k| f64::from(input[row * inner + k]) * f64::from(weights[column * inner + k]))
                .sum();
            f16::from_f64(sum).to_f32()
        })
        .collect();

    // Production loaders upload before constructing their execution context.
    let a = CudaBackend::from_slice(&input);
    let b = CudaBackend::from_slice(&weights);
    let mut out = CudaBackend::from_slice(&vec![sentinel; count + guard]);
    let mut copies = CudaBackend::from_slice(&vec![sentinel; 2 * count + 3 * guard]);
    let mut context = CudaBackend::new_context();
    CudaBackend::gemm(&mut context, &a, &b, &mut out, rows, columns, inner);
    CudaBackend::copy_slice(&mut context, &out, 0, &mut copies, guard, count);
    CudaBackend::scale_inplace(&mut context, &mut out, 0.5, count);
    CudaBackend::copy_slice(&mut context, &out, 0, &mut copies, count + 2 * guard, count);
    let complete = context
        .stream
        .record_event(None)
        .expect("record queued work");
    CudaBackend::sync(&mut context);
    // Check completion before the host copy, which could itself synchronize.
    assert!(
        complete.is_complete(),
        "Backend::sync returned before its queued work completed"
    );
    let mut expected_copies = vec![sentinel; 2 * count + 3 * guard];
    expected_copies[guard..guard + count].copy_from_slice(&expected);
    for (destination, value) in expected_copies[count + 2 * guard..2 * count + 2 * guard]
        .iter_mut()
        .zip(&expected)
    {
        *destination = value * 0.5;
    }
    // Dyadic inputs and these short dot products are exactly representable in
    // F16. Exact equality also rejects unwritten, nonfinite or reordered data.
    assert_eq!(
        CudaBackend::to_vec(&copies, expected_copies.len()),
        expected_copies
    );
    assert_eq!(
        &CudaBackend::to_vec(&out, count + guard)[count..],
        &[sentinel; 7]
    );

    // A following context must retain stream/scalar ownership for existing
    // allocations, without corrupting either earlier completed copy.
    drop(context);
    let mut context = CudaBackend::new_context();
    CudaBackend::scale_inplace(&mut context, &mut out, -2.0, count);
    CudaBackend::copy_slice(&mut context, &out, 0, &mut copies, guard, count);
    let complete = context
        .stream
        .record_event(None)
        .expect("record reused context");
    CudaBackend::sync(&mut context);
    assert!(complete.is_complete());
    for (destination, value) in expected_copies[guard..guard + count]
        .iter_mut()
        .zip(&expected)
    {
        *destination = -*value;
    }
    assert_eq!(
        CudaBackend::to_vec(&copies, expected_copies.len()),
        expected_copies
    );
    assert_eq!(
        &CudaBackend::to_vec(&out, count + guard)[count..],
        &[sentinel; 7]
    );
}
