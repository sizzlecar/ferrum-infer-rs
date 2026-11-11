//! CPU reference qualification of tentative GDN state, not speculative decoding.
mod fixture;

use ferrum_runtime::*;
use fixture::*;

fn config(rows: u64) -> Config {
    Config {
        rows,
        candidates: true,
        f32_hidden: false,
        log_rate: true,
        grouped: true,
    }
}
fn original_states() -> (Vec<u8>, Vec<u8>) {
    (
        encode(
            (0..CONV_BYTES / 2).map(|i| (i as f32 * 0.19).sin() * 0.2),
            ElementType::F16,
        ),
        encode(
            (0..DELTA_BYTES / 4).map(|i| (i as f32 * 0.27).cos() * 0.1),
            ElementType::F32,
        ),
    )
}
fn hidden(rows: usize, offset: usize, dtype: ElementType) -> Vec<u8> {
    encode(
        (offset * H..(offset + rows) * H).map(|i| (i as f32 * 0.31).sin() + 0.000123),
        dtype,
    )
}

#[test]
fn cpu_gdn_preprovisioned_candidates_match_serial_prefixes_prefill_and_selected_continuation() {
    let composition =
        ObservedComposition::create(id("device.cpu.gdn.candidates.numeric"), 4 << 20).unwrap();
    let host = host();
    for f32_hidden in [false, true] {
        for log_rate in [false, true] {
            for grouped in [false, true] {
                let c = Config {
                    f32_hidden,
                    log_rate,
                    grouped,
                    ..config(1)
                };
                let mut serial = session(
                    Config {
                        candidates: false,
                        ..c
                    },
                    &composition,
                    &host,
                );
                let mut single = session(c, &composition, &host);
                // One token plan handles boundary, verify-two and longer prefill work.
                let mut candidate =
                    session_preprovisioned(Config { rows: 5, ..c }, &composition, &host);
                for rows in [1_usize, 2, 5, 1] {
                    let candidate_config = Config {
                        rows: rows as u64,
                        ..c
                    };
                    let (original_conv, original_delta) = original_states();
                    let x = hidden(rows, 0, c.dtype());
                    let actual = run(
                        &mut candidate,
                        &host,
                        candidate_config,
                        &x,
                        &original_conv,
                        &original_delta,
                    );
                    assert_eq!(
                        bytes(&actual, C),
                        original_conv,
                        "read-only convolution was changed"
                    );
                    assert_eq!(
                        bytes(&actual, D),
                        original_delta,
                        "read-only recurrence was changed"
                    );
                    let conv = bytes(&actual, CC);
                    let delta = bytes(&actual, DD);
                    assert_eq!(conv.len(), 2 * CONV_BYTES);
                    assert_eq!(delta.len(), 2 * DELTA_BYTES);
                    let mut serial_conv = original_conv.clone();
                    let mut serial_delta = original_delta.clone();
                    let mut prior_conv = serial_conv.clone();
                    let mut prior_delta = serial_delta.clone();
                    let mut serial_hidden = Vec::new();
                    let row_bytes = H * c.dtype().size_bytes() as usize;
                    for row in 0..rows {
                        prior_conv.clone_from(&serial_conv);
                        prior_delta.clone_from(&serial_delta);
                        let expected = run(
                            &mut serial,
                            &host,
                            Config {
                                candidates: false,
                                ..c
                            },
                            &x[row * row_bytes..(row + 1) * row_bytes],
                            &serial_conv,
                            &serial_delta,
                        );
                        serial_conv = bytes(&expected, C).to_vec();
                        serial_delta = bytes(&expected, D).to_vec();
                        serial_hidden.extend_from_slice(bytes(&expected, Y));
                    }
                    assert_eq!(bytes(&actual, Y), serial_hidden);
                    assert_eq!(&conv[..CONV_BYTES], prior_conv);
                    assert_eq!(&delta[..DELTA_BYTES], prior_delta);
                    assert_eq!(&conv[CONV_BYTES..], serial_conv);
                    assert_eq!(&delta[DELTA_BYTES..], serial_delta);
                    assert_ne!(&conv[..CONV_BYTES], &conv[CONV_BYTES..]);
                    assert_ne!(&delta[..DELTA_BYTES], &delta[DELTA_BYTES..]);
                    // An independently chosen next row after either candidate
                    // tests convolution history AND recurrence rollback. Copying
                    // only a logical token index cannot satisfy these checks.
                    let next = hidden(1, 17, c.dtype());
                    for (slot, (expected_conv, expected_delta)) in
                        [(prior_conv, prior_delta), (serial_conv, serial_delta)]
                            .into_iter()
                            .enumerate()
                    {
                        let got = run(
                            &mut single,
                            &host,
                            c,
                            &next,
                            &conv[slot * CONV_BYTES..(slot + 1) * CONV_BYTES],
                            &delta[slot * DELTA_BYTES..(slot + 1) * DELTA_BYTES],
                        );
                        let expected = run(
                            &mut serial,
                            &host,
                            Config {
                                candidates: false,
                                ..c
                            },
                            &next,
                            &expected_conv,
                            &expected_delta,
                        );
                        assert_eq!(bytes(&got, Y), bytes(&expected, Y));
                        assert_eq!(&bytes(&got, CC)[..CONV_BYTES], expected_conv);
                        assert_eq!(&bytes(&got, DD)[..DELTA_BYTES], expected_delta);
                        assert_eq!(&bytes(&got, CC)[CONV_BYTES..], bytes(&expected, C));
                        assert_eq!(&bytes(&got, DD)[DELTA_BYTES..], bytes(&expected, D));
                    }
                    drop(actual);
                }
                close(candidate);
                close(single);
                close(serial);
                assert_eq!(composition.runtime().resident_bytes(), 0);
                assert_eq!(host.usage().unwrap().committed_bytes(), 0);
            }
        }
    }
}

#[test]
fn cpu_gdn_candidate_outputs_keep_their_host_owner_after_session_retirement() {
    let composition =
        ObservedComposition::create(id("device.cpu.gdn.candidates.retirement"), 4 << 20).unwrap();
    let host = host();
    let c = config(2);
    let mut s = session(c, &composition, &host);
    let (conv, delta) = original_states();
    let output = run(&mut s, &host, c, &hidden(2, 0, c.dtype()), &conv, &delta);
    let saved = bytes(&output, DD).to_vec();
    close(s);
    assert_eq!(composition.runtime().resident_bytes(), 0);
    assert!(host.usage().unwrap().allocated_bytes() >= 2 * DELTA_BYTES);
    assert_eq!(bytes(&output, DD), saved);
    drop(output);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn cpu_gdn_candidates_are_admitted_in_full_and_aliases_are_rejected_before_allocation() {
    let roomy = ObservedComposition::create(id("device.cpu.gdn.candidates.plan"), 4 << 20).unwrap();
    let component = compile(config(2), &roomy).unwrap();
    let memory = component.executable().execution_plan().payload().memory();
    let required = memory.static_bytes() + memory.minimum_runnable_request_bytes();
    let ordinary = compile(
        Config {
            candidates: false,
            ..config(2)
        },
        &roomy,
    )
    .unwrap();
    let original = ordinary.executable().execution_plan().payload().memory();
    assert!(
        required
            >= original.static_bytes()
                + original.minimum_runnable_request_bytes()
                + 2 * (CONV_BYTES + DELTA_BYTES) as u64
    );
    let small = ObservedComposition::create(id("device.cpu.gdn.candidates.capacity"), required - 1)
        .unwrap();
    assert!(matches!(
        compile(config(2), &small),
        Err(VNextError::PlanCapacityExceeded { .. })
    ));
    assert_eq!(small.runtime().resident_bytes(), 0);
    assert_eq!(small.runtime().peak_resident_bytes(), 0);

    let descriptor = gated_delta_candidate_scan_f16_contract().unwrap();
    let values = component.executable().execution_plan().payload().nodes()[0].values();
    descriptor
        .descriptor()
        .validate_resolved_bindings(values)
        .unwrap();
    // Real compiled bindings supply all typed geometry. Only corrupt the
    // proposed physical placement; no allocation permit is manufactured.
    for (destination, aliased) in [(10, 0), (11, 8), (12, 9), (11, 10)] {
        let mut invalid = values.to_vec();
        let output = &values[destination];
        let source = &values[aliased].storage().components()[0];
        let storage = ResolvedValueStorage::single(
            source.resource_id().clone(),
            source.offset_bytes(),
            output.tensor().minimum_storage_bytes().unwrap(),
            output.tensor().element_type(),
        )
        .unwrap();
        invalid[destination] = ResolvedValueBinding::new(
            output.value_id().clone(),
            output.role(),
            output.ordinal(),
            output.tensor().clone(),
            output.access(),
            output.alias().clone(),
            output.usage(),
            output.weight().cloned(),
            storage,
        )
        .unwrap();
        assert!(descriptor
            .descriptor()
            .validate_resolved_bindings(&invalid)
            .is_err());
    }
    assert_eq!(roomy.runtime().resident_bytes(), 0);
    assert_eq!(roomy.runtime().peak_resident_bytes(), 0);
}

#[test]
fn cpu_gdn_elastic_idle_fragmentation_preserves_floor_and_reports_pressure() {
    let composition =
        ObservedComposition::create(id("device.cpu.gdn.candidates.fragmented"), 4 << 20).unwrap();
    let host = host();
    let c = config(5);
    // Retain ordinary demand-driven initialization to record the unsupported
    // idle 2-row -> 5-row packing transition; preprovisioning must not hide it.
    let mut session = session(c, &composition, &host);
    let (conv, delta) = original_states();
    for rows in [1_usize, 2] {
        let actual = run(
            &mut session,
            &host,
            Config {
                rows: rows as u64,
                ..c
            },
            &hidden(rows, 0, c.dtype()),
            &conv,
            &delta,
        );
        drop(actual);
    }
    assert_idle_fragmented_growth_waits(&mut session, c.rows);
    close(session);
    assert_eq!(composition.runtime().resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
