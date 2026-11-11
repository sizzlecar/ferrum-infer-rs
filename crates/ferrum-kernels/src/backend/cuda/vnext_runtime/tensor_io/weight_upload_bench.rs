//! Isolate pageable versus page-locked source transfer cost. This is a native
//! transfer diagnostic, not a managed allocator or whole-model speed claim.
use cudarc::driver::CudaContext;
use serde_json::json;
use std::time::Instant;

#[test]
#[ignore = "requires exclusive CUDA; includes full transfer validation, not model performance"]
fn pageable_and_pinned_weight_uploads_preserve_all_bytes_on_cuda() {
    const GUARD: usize = 256;
    let context = CudaContext::new(0).unwrap();
    let stream = context.new_stream().unwrap();

    // The larger range is the existing streaming source's physical ceiling;
    // the smaller range also checks transfers that do not fill that slot.
    for bytes in [16 << 20, 64 << 20] {
        let allocated = Instant::now();
        let mut pageable = vec![0_u8; bytes];
        let pageable_allocate_us = allocated.elapsed().as_micros();
        for (index, value) in pageable.iter_mut().enumerate() {
            *value = (index.wrapping_mul(37) ^ (index >> 11)) as u8;
        }
        let allocated = Instant::now();
        // SAFETY: initialize the complete u8 allocation before reading it or
        // submitting it. The pinned owner retains its stream completion event.
        let mut pinned = unsafe { context.alloc_pinned::<u8>(bytes) }.unwrap();
        let pinned_allocate_us = allocated.elapsed().as_micros();
        let initialized = Instant::now();
        pinned.as_mut_slice().unwrap().copy_from_slice(&pageable);
        let pinned_initialize_us = initialized.elapsed().as_micros();
        let mut device = stream
            .clone_htod(&vec![0xa7_u8; bytes + 2 * GUARD])
            .unwrap();
        stream.synchronize().unwrap();

        let mut observations = Vec::new();
        for repetition in 0..8 {
            // Pair and alternate order; retain cold and warm samples alike.
            for use_pinned in if repetition % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mut destination = device.slice_mut(GUARD..GUARD + bytes);
                let started = Instant::now();
                if use_pinned {
                    stream.memcpy_htod(&pinned, &mut destination).unwrap();
                } else {
                    stream
                        .memcpy_htod(pageable.as_slice(), &mut destination)
                        .unwrap();
                }
                let submit_us = started.elapsed().as_micros();
                stream.synchronize().unwrap();
                let completion_us = started.elapsed().as_micros();
                drop(destination);

                // Readback and comparison are outside the timed transfer and
                // follow its actual completion, including both guard regions.
                let actual = stream.clone_dtoh(&device).unwrap();
                assert!(actual[..GUARD].iter().all(|&byte| byte == 0xa7));
                assert_eq!(&actual[GUARD..GUARD + bytes], pageable.as_slice());
                assert!(actual[GUARD + bytes..].iter().all(|&byte| byte == 0xa7));
                observations.push(json!({
                    "repetition": repetition,
                    "source": if use_pinned {"pinned_write_combined"} else {"pageable"},
                    "submit_us": submit_us,
                    "completion_us": completion_us,
                }));
            }
        }
        println!(
            "{}",
            json!({
                "kind":"weight_upload_diagnostic", "bytes":bytes,
                "pageable_allocate_zero_us":pageable_allocate_us,
                "pinned_allocate_us":pinned_allocate_us,
                "pinned_copy_initialization_us":pinned_initialize_us,
                "observations":observations,
                "limits":[
                    "native CUDA transfer only; no source conversion, model execution or SLO claim",
                    "pinned arm uses cuMemHostAlloc write-combined storage, not a registered mmap",
                    "readback touches the full payload between samples; no cache-state equivalence claim",
                    "managed Host quotas, reusable source slots and driver failure ownership are not implemented by this diagnostic"
                ]
            })
        );
    }
}
