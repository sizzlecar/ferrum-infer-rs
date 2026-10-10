//! Synthetic transfer-only diagnostic. This does not cache production commands
//! or replace fresh invocation authority checks. No numerical kernel executes.

use super::{
    coalesce_program_binding_transfers, CudaProgramBindingTransfer, CudaProgramBindingWrite,
};
use serde_json::json;
use std::time::Instant;

mod fixture;
use fixture::Fixture;

type Result<T> = std::result::Result<T, String>;
const CANARY: u8 = 0xd3;

#[derive(Clone, Copy, Debug)]
enum Route {
    BoxDirect,
    PinnedDirect,
    PinnedGraph,
}

impl Route {
    const ALL: [Self; 3] = [Self::BoxDirect, Self::PinnedDirect, Self::PinnedGraph];
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CopyShape {
    offset: u64,
    pitch: u64,
    width: usize,
    rows: usize,
}

impl From<&CudaProgramBindingTransfer> for CopyShape {
    fn from(transfer: &CudaProgramBindingTransfer) -> Self {
        Self {
            offset: transfer.destination_offset_bytes,
            pitch: transfer.destination_stride_bytes,
            width: transfer.row_bytes,
            rows: transfer.row_count,
        }
    }
}

#[derive(Clone)]
struct Geometry {
    participants: usize,
    // Original logical writes, independent of the coalescer's packed layout.
    windows: Vec<(usize, usize)>,
    arena_bytes: usize,
}

impl Geometry {
    fn new(participants: usize) -> Self {
        assert!(participants > 0);
        let mut windows = Vec::new();
        let mut base = 64;
        // Two synthetic copies of each provider layout, not a model's node or
        // transfer count. GDN StateBindingLayout has 16 bytes per participant.
        // Causal VllmBlocks16 at max_context=2048 has 24 control bytes plus
        // align16(128 * sizeof(u64)) = 1024 table bytes: slot stride 1048.
        // Only live prefixes for frontiers 64/80/96 are written (56/64/72 B).
        for _ in 0..2 {
            for participant in 0..participants {
                windows.push((base + participant * 16, 16));
            }
            base += participants * 16 + 64;
            for participant in 0..participants {
                let entries = 4 + (participant / 4) % 3;
                windows.push((base + participant * 1048, 24 + entries * 8));
            }
            base += participants * 1048 + 64;
        }
        Self {
            participants,
            windows,
            arena_bytes: base,
        }
    }

    fn byte(wave: u64, window: usize, byte: usize) -> u8 {
        // All live bytes change between consecutive waves, including bytes
        // which represent addresses in a real provider (not dereferenced here).
        ((wave.wrapping_mul(29) + window as u64 * 17 + byte as u64 * 7) % 251) as u8
    }

    fn prepare(&self, wave: u64) -> Result<Vec<CudaProgramBindingTransfer>> {
        let writes = self
            .windows
            .iter()
            .enumerate()
            .map(|(window, &(offset, size))| {
                let payload = (0..size)
                    .map(|byte| Self::byte(wave, window, byte))
                    .collect::<Vec<_>>()
                    .into_boxed_slice();
                CudaProgramBindingWrite::new(offset as u64, payload)
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        coalesce_program_binding_transfers(writes, self.arena_bytes as u64)
            .map_err(|error| error.to_string())
    }

    fn expected(&self, wave: u64) -> Vec<u8> {
        let mut expected = vec![CANARY; self.arena_bytes];
        for (window, &(offset, size)) in self.windows.iter().enumerate() {
            for byte in 0..size {
                expected[offset + byte] = Self::byte(wave, window, byte);
            }
        }
        expected
    }
}

#[derive(Default)]
struct Measurement {
    net_ns: u128,
    host_api_ns: u128,
    stream_interval_ns: f64,
}

#[test]
#[ignore = "real CUDA sparse-copy graph lifetime and full-byte oracle; exclusive GPU"]
fn sparse_upload_prelude_fresh_bytes_and_lifetimes() {
    for participants in [8, 32] {
        let geometry = Geometry::new(participants);
        let mut fixture = Fixture::new(geometry.clone()).unwrap();
        // Capture/instantiate/upload may not execute its copies.
        assert_eq!(
            fixture.readback().unwrap(),
            vec![CANARY; geometry.arena_bytes]
        );
        let mut wave = 1;
        for _ in 0..4 {
            for route in Route::ALL {
                fixture.run(route, wave, None).unwrap();
                assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                wave += 1;
            }
        }

        // A host-injected failure after one real asynchronous enqueue must
        // fence before freeing pageable payloads or allowing pinned mutation.
        // This is cleanup coverage, not a simulated CUDA driver failure.
        for route in [Route::BoxDirect, Route::PinnedDirect] {
            assert!(fixture.run(route, wave, Some(1)).is_err());
            wave += 1;
            fixture.run(Route::PinnedGraph, wave, None).unwrap();
            assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
            wave += 1;
        }

        // A changed sparse shape is rejected before graph launch or pin writes;
        // it cannot replay stale addresses or overwrite a larger live prefix.
        let before = fixture.readback().unwrap();
        assert!(fixture.reject_changed_shape(wave).is_err());
        assert_eq!(fixture.readback().unwrap(), before);
        fixture.abort_capture_after_copy().unwrap();
        assert_eq!(fixture.readback().unwrap(), before);
        fixture.run(Route::PinnedGraph, wave, None).unwrap();
        assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));

        // Drop while a real replay is in flight. The owner fences and destroys
        // the graph before releasing its pinned sources and device allocation.
        // Retain just the destination for an independent post-drop observation.
        let (stream, destination) = fixture.destination_observer();
        fixture.launch_for_drop(wave + 1).unwrap();
        drop(fixture);
        assert_eq!(
            stream.clone_dtoh(destination.as_ref()).unwrap(),
            geometry.expected(wave + 1)
        );
        println!(
            "sparse_upload_prelude_correctness {}",
            json!({
                "participants": participants, "arena_bytes": geometry.arena_bytes,
                "full_byte_and_gap_oracle": true, "changed_payload_every_wave": true,
                "partial_enqueue_cleanup": true, "capture_abort_cleanup": true,
                "inflight_drop_cleanup": true, "driver_failure_injected": false,
            })
        );
    }
}

#[test]
#[ignore = "balanced pageable/pinned/direct/captured sparse transfer net-cost diagnostic; exclusive GPU"]
fn sparse_upload_prelude_paired_net_cost() {
    const WARM_WAVES: usize = 4;
    const QUADS: usize = 8;
    const WAVES_PER_ARM: usize = 16;
    println!(
        "sparse_upload_prelude_plan {}",
        json!({
            "scope": "synthetic sparse transfer prelude; not whole-model or serving evidence",
            "participants": [8, 32], "synthetic_gdn_nodes": 2, "synthetic_causal_nodes": 2,
            "causal_slot_bytes": 1048, "live_causal_bytes": [56, 64, 72],
            "warm_waves_per_route": WARM_WAVES, "quads_per_comparison": QUADS,
            "waves_per_arm": WAVES_PER_ARM,
            "comparisons": ["BoxDirect/PinnedGraph", "PinnedDirect/PinnedGraph"],
            "order": "even quad A B B A; odd quad B A A B",
            "net_scope": "fresh Box payload fill + production sparse coalescer + validation + pinned staging if applicable + actual commands + events + fence + per-wave payload retirement",
            "stream_interval_scope": "prebuilt start after host fill, stop after last copy; includes host API gaps, not pure DMA or SM busy",
            "readback": "full arena and gap canary after each wave, outside all measured intervals",
            "setup": "device common setup and pinned/graph setup reported separately; never hidden in an amortized net value",
            "geometry": "fixed for a case; payload changes every wave; same actual destination for all routes",
            "authority_scope": "fixture-owned allocation only; no production authority/cache implementation",
        })
    );
    for participants in [8, 32] {
        let geometry = Geometry::new(participants);
        let mut fixture = Fixture::new(geometry.clone()).unwrap();
        fixture.print_setup();
        let mut wave = 1;
        for route in Route::ALL {
            for _ in 0..WARM_WAVES {
                fixture.run(route, wave, None).unwrap();
                assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                wave += 1;
            }
        }
        for direct in [Route::BoxDirect, Route::PinnedDirect] {
            for quad in 0..QUADS {
                let quad_first_wave = wave;
                let order = if quad % 2 == 0 {
                    [direct, Route::PinnedGraph, Route::PinnedGraph, direct]
                } else {
                    [Route::PinnedGraph, direct, direct, Route::PinnedGraph]
                };
                for (position, route) in order.into_iter().enumerate() {
                    // Paired arms use exactly the same changing payload series.
                    // Repeating the series still changes the preceding arm's
                    // final payload, so every replay must refresh device bytes.
                    wave = quad_first_wave;
                    let first_wave = wave;
                    let mut total = Measurement::default();
                    for _ in 0..WAVES_PER_ARM {
                        let measured = fixture.run(route, wave, None).unwrap();
                        total.net_ns += measured.net_ns;
                        total.host_api_ns += measured.host_api_ns;
                        total.stream_interval_ns += measured.stream_interval_ns;
                        assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                        wave += 1;
                    }
                    println!(
                        "sparse_upload_prelude_measurement {}",
                        json!({
                            "participants": participants, "comparison": format!("{direct:?}/PinnedGraph"),
                            "quad": quad, "position": position, "route": format!("{route:?}"),
                            "waves": WAVES_PER_ARM, "first_wave": first_wave,
                            "net_ns": total.net_ns, "host_api_ns": total.host_api_ns,
                            "stream_interval_ns": total.stream_interval_ns,
                        })
                    );
                }
            }
        }
    }
}
