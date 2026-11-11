use super::*;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};

fn host(capacity: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            capacity,
            HostMemoryClassLimits {
                input_preprocess_bytes: capacity,
                output_bytes: capacity,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn plan() -> PH3ImageVaeTilePlan {
    PH3ImageVaeTilePlan::new(
        12,
        12,
        2,
        PH3ImageVaeSpatialPolicy::Tiled {
            tile_pixels: 8,
            minimum_overlap_pixels: 4,
            max_tiles: 4,
        },
    )
    .unwrap()
}
fn output(host: &HostMemoryAccount, values: &[f32]) -> ManagedHostBytes {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::Output, values.len() * 4)
        .unwrap();
    for (dst, value) in allocation.as_mut().chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&value.to_le_bytes());
    }
    allocation.freeze()
}
fn tile(host: &HostMemoryAccount, index: usize, batch: usize, channels: usize) -> ManagedHostBytes {
    let values = (0..batch)
        .flat_map(|b| {
            (0..channels).flat_map(move |c| {
                std::iter::repeat_n((b * 100 + c * 1000 + index * 10) as f32, 16)
            })
        })
        .collect::<Vec<_>>();
    output(host, &values)
}
fn expected() -> [f32; 36] {
    [
        0., 0., 0., 5., 10., 10., 0., 0., 0., 5., 10., 10., 0., 0., 20., 15., 10., 10., 10., 10.,
        20., 20., 20., 20., 20., 20., 20., 25., 30., 30., 20., 20., 20., 25., 30., 30.,
    ]
}
#[test]
fn raw_encoder_neighbours_preserve_literal_cross_seams_batch_channels_and_scaled_geometry() {
    let host = host(1 << 20);
    let plan = plan();
    let peak = PH3ImageVaeMomentCanvas::output_peak(&host, &plan, 2, 4).unwrap();
    let mut canvas = PH3ImageVaeMomentCanvas::new(plan, host.clone(), 2, 4).unwrap();
    let spec = f32_spec(&[2, 4, 4, 4]).unwrap();
    for i in 0..4 {
        let raw = tile(&host, i, 2, 4);
        let original = raw.bytes().to_vec();
        canvas.push(i, &spec, &raw).unwrap();
        assert_eq!(raw.bytes(), original);
        assert!(std::ptr::eq(
            canvas.raw_row[i % 2].as_ref().unwrap().bytes().as_ptr(),
            raw.bytes().as_ptr()
        ));
        assert!(host.usage().unwrap().committed_bytes() <= peak);
    }
    let (spec, data) = canvas.finish().unwrap();
    assert_eq!(spec.dimensions(), [2, 4, 6, 6]);
    for b in 0..2 {
        for c in 0..4 {
            for (i, wanted) in expected().into_iter().enumerate() {
                assert_eq!(
                    read(data.bytes(), ((b * 4 + c) * 36 + i) * 4),
                    wanted + (b * 100 + c * 1000) as f32
                );
            }
        }
    }
    // Recursive decoder blending produces zero here; the original encoder
    // uses its raw left neighbour and must retain twenty instead.
    assert_eq!(read(data.bytes(), (2 * 6 + 2) * 4), 20.);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        host.backing_bytes(data.len()).unwrap()
    );
    drop(data);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn aligned_capacity_failure_keeps_raw_neighbours_and_committed_canvas_for_retry() {
    let probe = host(1 << 20);
    let peak = PH3ImageVaeMomentCanvas::output_peak(&probe, &plan(), 1, 1).unwrap();
    let account = host(peak);
    let mut canvas = PH3ImageVaeMomentCanvas::new(plan(), account.clone(), 1, 1).unwrap();
    let spec = f32_spec(&[1, 1, 4, 4]).unwrap();
    for i in 0..2 {
        canvas.push(i, &spec, &tile(&account, i, 1, 1)).unwrap();
    }
    let raw = tile(&account, 2, 1, 1);
    let before = canvas.canvas.as_ref().to_vec();
    let previous = canvas
        .raw_row
        .iter()
        .map(|v| v.as_ref().unwrap().bytes().as_ptr())
        .collect::<Vec<_>>();
    let blocker = account.allocate_zeroed(HostMemoryClass::Output, 1).unwrap();
    let blocked = account.usage().unwrap().committed_bytes();
    assert!(canvas.push(2, &spec, &raw).is_err());
    assert_eq!(canvas.completed_tiles(), 2);
    assert_eq!(canvas.canvas.as_ref(), before);
    for (i, pointer) in previous.into_iter().enumerate() {
        assert_eq!(
            canvas.raw_row[i].as_ref().unwrap().bytes().as_ptr(),
            pointer
        );
    }
    assert_eq!(account.usage().unwrap().committed_bytes(), blocked);
    drop(blocker);
    canvas.push(2, &spec, &raw).unwrap();
    drop(raw);
    canvas.push(3, &spec, &tile(&account, 3, 1, 1)).unwrap();
    let (_, data) = canvas.finish().unwrap();
    assert_eq!(
        data.bytes()
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
        expected()
    );
    drop(data);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn bad_order_foreign_owner_or_nonfinite_moments_cannot_publish_partial_output() {
    let account = host(1 << 20);
    let mut canvas = PH3ImageVaeMomentCanvas::new(plan(), account.clone(), 1, 1).unwrap();
    let spec = f32_spec(&[1, 1, 4, 4]).unwrap();
    let raw = tile(&account, 0, 1, 1);
    assert!(canvas.push(1, &spec, &raw).is_err());
    let foreign = host(1 << 20);
    assert!(canvas.push(0, &spec, &tile(&foreign, 0, 1, 1)).is_err());
    let mut poisoned = [0.; 16];
    poisoned[15] = f32::NAN;
    assert!(canvas.push(0, &spec, &output(&account, &poisoned)).is_err());
    assert_eq!(canvas.completed_tiles(), 0);
    canvas.push(0, &spec, &raw).unwrap();
    drop(raw);
    assert!(canvas.finish().is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
