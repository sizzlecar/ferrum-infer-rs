//! Submission-mechanism diagnostic, preserving sparse destination bytes.
//! Fixture ownership checks are not production Core authority validation.

use super::{
    coalesce_program_binding_transfers, CudaProgramBindingTransfer, CudaProgramBindingWrite,
};
use serde_json::json;
use std::time::Instant;

mod fixture;
mod packet;
use fixture::Fixture;
use packet::Packet;

type Result<T> = std::result::Result<T, String>;
const CANARY: u8 = 0xd3;
// The physical provider row retains the model's 262144-token capacity.
// Only the live prefix is uploaded, bounded by the service's 2048 tokens.
const CAUSAL_SLOT_BYTES: usize = 24 + (262144 / 16) * 8;

#[derive(Clone, Copy, Debug)]
enum Route {
    PageableDirect,
    PageableScatter,
    PinnedDirect,
    PinnedScatter,
}

impl Route {
    fn pinned(self) -> bool {
        matches!(self, Self::PinnedDirect | Self::PinnedScatter)
    }

    fn scatter(self) -> bool {
        matches!(self, Self::PageableScatter | Self::PinnedScatter)
    }
}

#[derive(Clone)]
struct Geometry {
    participants: usize,
    gdn_nodes: usize,
    causal_nodes: usize,
    base_pages: usize,
    arena_bytes: usize,
    packet_capacity: usize,
}

impl Geometry {
    fn new(participants: usize, gdn_nodes: usize, causal_nodes: usize, base_pages: usize) -> Self {
        assert!(participants > 0 && gdn_nodes + causal_nodes > 0);
        assert!((1..=126).contains(&base_pages));
        let arena_bytes = 64
            + gdn_nodes * (participants * 16 + 64)
            + causal_nodes * (participants * CAUSAL_SLOT_BYTES + 64);
        // The coalescer cannot emit more transfers than original writes.
        let max_writes = participants * (gdn_nodes + causal_nodes);
        let max_live = participants * (gdn_nodes * 16 + causal_nodes * (24 + (base_pages + 2) * 8));
        Self {
            participants,
            gdn_nodes,
            causal_nodes,
            base_pages,
            arena_bytes,
            packet_capacity: 40 * max_writes + max_live,
        }
    }

    fn windows(&self, wave: u64) -> Vec<(usize, usize, usize)> {
        let mut windows = Vec::new();
        let mut base = 64;
        for node in 0..self.gdn_nodes + self.causal_nodes {
            let causal = node >= self.gdn_nodes;
            let stride = if causal { CAUSAL_SLOT_BYTES } else { 16 };
            for slot in 0..self.participants {
                // Participant reorder and varying live prefixes change every
                // wave. The physical row keeps model capacity; the live
                // prefix stays within the declared 2048-token service limit.
                let participant = (slot + wave as usize % self.participants) % self.participants;
                let pages = self.base_pages + ((participant / 4 + wave as usize) % 3);
                let width = if causal { 24 + pages * 8 } else { 16 };
                windows.push((
                    base + slot * stride,
                    width,
                    node * self.participants + participant,
                ));
            }
            base += self.participants * stride + 64;
        }
        windows
    }

    fn byte(wave: u64, source: usize, byte: usize) -> u8 {
        ((wave.wrapping_mul(29) + source as u64 * 17 + byte as u64 * 7) % 251) as u8
    }

    fn prepare(&self, wave: u64) -> Result<Vec<CudaProgramBindingTransfer>> {
        let writes = self
            .windows(wave)
            .into_iter()
            .map(|(offset, size, source)| {
                let payload = (0..size)
                    .map(|byte| Self::byte(wave, source, byte))
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
        self.apply_expected(&mut expected, wave);
        expected
    }

    fn apply_expected(&self, expected: &mut [u8], wave: u64) {
        for (offset, size, source) in self.windows(wave) {
            for byte in 0..size {
                expected[offset + byte] = Self::byte(wave, source, byte);
            }
        }
    }
}

#[derive(Default)]
struct Measurement {
    net_ns: u128,
    host_api_ns: u128,
    stream_interval_ns: f64,
    transfers: usize,
    live_bytes: usize,
    upload_bytes: usize,
}

#[test]
#[ignore = "CUDA sparse byte-scatter oracle and asynchronous lifetime; exclusive GPU"]
fn compact_upload_scatter_fresh_bytes_and_lifetimes() {
    for participants in [8, 32] {
        let geometry = Geometry::new(participants, 2, 2, 126);
        let mut fixture = Fixture::new(geometry.clone()).unwrap();
        let mut wave = 1;
        for route in [
            Route::PageableDirect,
            Route::PageableScatter,
            Route::PinnedDirect,
            Route::PinnedScatter,
        ] {
            for _ in 0..4 {
                fixture.reset_destination().unwrap();
                fixture.run(route, wave, false).unwrap();
                assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                wave += 1;
            }
        }
        // Without resetting, a shorter prefix must preserve the previously
        // written tail, not merely untouched canary bytes.
        for route in [Route::PageableScatter, Route::PinnedScatter] {
            fixture.reset_destination().unwrap();
            fixture.run(route, 1, false).unwrap();
            let mut expected = geometry.expected(1);
            assert!(geometry
                .windows(1)
                .iter()
                .zip(geometry.windows(2))
                .any(|((_, before, _), (_, after, _))| after < *before));
            fixture.run(route, 2, false).unwrap();
            geometry.apply_expected(&mut expected, 2);
            assert_eq!(fixture.readback().unwrap(), expected);
        }
        // Reset is deliberately outside the measured prelude.
        for route in [Route::PageableScatter, Route::PinnedScatter] {
            fixture.reset_destination().unwrap();
            let before = fixture.readback().unwrap();
            assert!(fixture.run(route, wave, true).is_err());
            assert_eq!(fixture.readback().unwrap(), before);
            wave += 1;
            fixture.run(route, wave, false).unwrap();
            assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
            wave += 1;
        }
        fixture
            .check_unknown_completion_rejects_before_staging(wave)
            .unwrap();
        fixture.reset_destination().unwrap();
        let (stream, destination) = fixture.destination_observer();
        fixture.launch_pinned_scatter_for_drop(wave).unwrap();
        drop(fixture);
        assert_eq!(
            stream.clone_dtoh(destination.as_ref()).unwrap(),
            geometry.expected(wave)
        );
        println!(
            "compact_upload_scatter_correctness {}",
            json!({
                "participants": participants, "arena_bytes": geometry.arena_bytes,
                "full_byte_and_hole_oracle": true, "participant_reorder_and_live_prefix_changes": true,
                "failure_after_real_h2d_before_scatter": true, "inflight_scatter_drop": true,
                "unknown_completion_state_injected": true, "real_driver_fault_injected": false,
            })
        );
    }
}

#[test]
#[ignore = "paired complete sparse upload versus single H2D plus scatter net cost; exclusive GPU"]
fn compact_upload_scatter_paired_net_cost() {
    const WARM: usize = 8;
    const QUADS: usize = 8;
    const WAVES: usize = 16;
    let pairs = [
        (Route::PageableDirect, Route::PageableScatter),
        (Route::PinnedDirect, Route::PinnedScatter),
    ];
    println!(
        "compact_upload_scatter_plan {}",
        json!({
            "scope": "synthetic submission mechanism; no Core authority or serving performance claim",
            "warm_waves_per_route": WARM, "quads_per_pair": QUADS, "waves_per_arm": WAVES,
            "order": "alternating ABBA/BAAB; identical fresh input series within each quad",
            "geometry": "2/2 small and 48/16 production node-count layout; physical causal pitch 131096B, service 2048 tokens, live table only",
            "net_scope": "fresh payload fill + production coalescer + common bounds/overlap validation + descriptor/packet packing where used + matched staging + all submissions/events + terminal fence + transient retirement",
            "stream_interval_scope": "events after staging to last copy/scatter; includes host enqueue gaps, not pure DMA or SM busy",
            "setup": "context/PTX/module/device/pin/event allocation separately reported, no per-wave allocation excluded",
            "oracle": "full destination bytes/holes each wave, reset/readback outside measured intervals",
            "authority_scope": "fixture-owned arena; unchanged Core checks would remain during product integration",
        })
    );
    for participants in [8, 32] {
        for (gdn, causal, pages) in [(2, 2, 6), (48, 16, 6), (48, 16, 64)] {
            let geometry = Geometry::new(participants, gdn, causal, pages);
            let mut fixture = Fixture::new(geometry.clone()).unwrap();
            fixture.print_setup();
            let mut wave = 1;
            for (a, b) in pairs {
                for route in [a, b] {
                    for _ in 0..WARM {
                        fixture.reset_destination().unwrap();
                        fixture.run(route, wave, false).unwrap();
                        assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                        wave += 1;
                    }
                }
                for quad in 0..QUADS {
                    let first = wave;
                    let order = if quad % 2 == 0 {
                        [a, b, b, a]
                    } else {
                        [b, a, a, b]
                    };
                    for (position, route) in order.into_iter().enumerate() {
                        wave = first;
                        let mut total = Measurement::default();
                        for _ in 0..WAVES {
                            fixture.reset_destination().unwrap();
                            let m = fixture.run(route, wave, false).unwrap();
                            total.net_ns += m.net_ns;
                            total.host_api_ns += m.host_api_ns;
                            total.stream_interval_ns += m.stream_interval_ns;
                            total.transfers += m.transfers;
                            total.live_bytes += m.live_bytes;
                            total.upload_bytes += m.upload_bytes;
                            assert_eq!(fixture.readback().unwrap(), geometry.expected(wave));
                            wave += 1;
                        }
                        println!(
                            "compact_upload_scatter_measurement {}",
                            json!({
                                "participants": participants, "gdn_nodes": gdn, "causal_nodes": causal,
                                "base_pages": pages, "quad": quad, "position": position,
                                "route": format!("{route:?}"), "waves": WAVES, "first_wave": first,
                                "net_ns": total.net_ns, "host_api_ns": total.host_api_ns,
                                "stream_interval_ns": total.stream_interval_ns,
                                "sparse_transfers": total.transfers, "live_bytes": total.live_bytes,
                                "uploaded_bytes": total.upload_bytes,
                            })
                        );
                    }
                }
            }
        }
    }
}
