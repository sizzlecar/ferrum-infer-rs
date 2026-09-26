use super::*;

#[test]
fn native_labels_are_bytes_and_do_not_parse_unrelated_counters_as_memory() {
    // Shape of real macOS time -l output; distinct values catch swapped columns.
    let report = b"  4.68 real 2.79 user 0.44 sys\n 477102080 maximum resident set size\n 0 swaps\n 523634920 peak memory footprint\n";
    assert_eq!(
        parse_peaks(report).unwrap(),
        NativePeaks {
            footprint: Some(523_634_920),
            rss: Some(477_102_080)
        }
    );
}

#[test]
fn duplicate_fields_are_ambiguous_even_if_values_agree() {
    for label in ["maximum resident set size", "peak memory footprint"] {
        let report = format!("1 {label}\n1 {label}\n");
        assert!(parse_peaks(report.as_bytes())
            .unwrap_err()
            .0
            .contains("duplicate"));
    }
}

#[test]
fn native_field_units_signs_fractional_and_overflow_values_are_rejected() {
    for value in [
        "1 KiB",
        "1 B",
        "1,000",
        "-1",
        "+1",
        "1.5",
        "NaN",
        "18446744073709551616",
        "",
    ] {
        let report = format!("{value} peak memory footprint\n");
        assert!(parse_peaks(report.as_bytes()).is_err(), "{report}");
    }
    assert!(parse_peaks(b"1 maximum resident set size KiB\n").is_err());
    assert!(parse_peaks(b"1peak memory footprint\n").is_err());
    assert!(parse_peaks(&[0xff]).is_err());
    assert!(parse_peaks(&vec![b' '; MAX_TIME_REPORT_BYTES + 1]).is_err());
}

#[test]
fn absent_fields_remain_absent_not_zero() {
    assert_eq!(
        parse_peaks(b"0.5 real 0 user 0 sys\n").unwrap(),
        NativePeaks::default()
    );
    assert_eq!(
        parse_peaks(b"4 peak memory footprint\n").unwrap(),
        NativePeaks {
            footprint: Some(4),
            rss: None
        }
    );
}

#[test]
fn exit_status_is_original_scalar_not_a_success_statement() {
    assert_eq!(parse_exit(b"0\n").unwrap(), 0);
    assert_eq!(parse_exit(b"130\n").unwrap(), 130);
    for bytes in [b"success".as_slice(), b"0\n1", b"-1", b"256", b""] {
        assert!(parse_exit(bytes).is_err());
    }
}

#[test]
fn sampled_policy_wire_is_unchanged_and_lifetime_is_explicit_strict() {
    let sampled = SampledMemoryPolicy {
        window: "measure".into(),
        interval_ms: 100,
        max_sample_gap_ns: 200_000_000,
    };
    let old = serde_json::to_vec(&sampled).unwrap();
    let new: OsFootprintPolicy = serde_json::from_slice(&old).unwrap();
    assert_eq!(serde_json::to_vec(&new).unwrap(), old);
    let lifetime: OsFootprintPolicy =
        serde_json::from_str(r#"{"kind":"macos_time_l_v1"}"#).unwrap();
    assert_eq!(
        lifetime.measurement(),
        MemoryMeasurement::ProcessPeakPhysicalFootprint
    );
    assert!(lifetime.sampled().is_none());
    for input in [
        r#"{"kind":"macos_time_l_v2"}"#,
        r#"{"kind":"macos_time_l_v1","interval_ms":100}"#,
        r#"{"window":"measure","interval_ms":100,"max_sample_gap_ns":1,"complete":true}"#,
    ] {
        assert!(serde_json::from_str::<OsFootprintPolicy>(input).is_err());
    }
}
