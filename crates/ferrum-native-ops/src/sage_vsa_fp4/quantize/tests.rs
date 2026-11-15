use super::*;
use crate::sage_vsa_fp4::SageVsaFp4RequestV1;

fn device() -> Device {
    Device {
        ordinal: 0,
        architecture: 1200,
        multiprocessors: 1,
        maximum_dynamic_shared_bytes: 128 << 10,
    }
}
fn plan() -> Plan {
    let request = Request::new(2, 3, 128, 384, device()).unwrap();
    Plan {
        request,
        abi: 1,
        size: size_of::<Plan>() as u32,
        query_grid_x: 1,
        key_grid_x: 3,
        grid_y: 2,
        grid_z: 3,
        threads: 1024,
        launch_count: 3,
        value_shared_bytes: 32768,
        bytes: request.buffer_sizes().unwrap(),
        ..Default::default()
    }
}

#[test]
fn quantized_spans_match_actual_sparse_consumer_but_inputs_remain_sequence_major() {
    let request = Request::new(2, 3, 128, 384, device()).unwrap();
    assert_eq!(request.query_strides, [49152, 128, 384]);
    assert_eq!(request.key_strides, [147456, 128, 384]);
    let produced = request.buffer_sizes().unwrap();
    let consumed = SageVsaFp4RequestV1::new(2, 3, 1, 3, 0.125, device())
        .unwrap()
        .buffer_sizes()
        .unwrap();
    assert_eq!(
        [
            produced.query,
            produced.key,
            produced.value,
            produced.query_scales,
            produced.key_scales,
            produced.value_scales
        ],
        [
            consumed.query,
            consumed.key,
            consumed.value,
            consumed.query_scales,
            consumed.key_scales,
            consumed.value_scales
        ]
    );
    assert_eq!(produced.query_input, 196608);
    assert_eq!(produced.key_input, 589824);
    assert_eq!(produced.value_input, produced.key_input);
}

#[test]
fn quantization_rejects_implicit_transposes_partial_tiles_and_native_integer_overflow() {
    let mut request = Request::new(2, 3, 128, 384, device()).unwrap();
    request.key_strides.swap(1, 2);
    assert_eq!(request.validate(), Err(Error::UnsupportedGeometry));
    assert!(Request::new(1, 1, 128, 129, device()).is_err());
    assert!(Request::new(1, 0, 128, 128, device()).is_err());
    let largest_rows = (i32::MAX as u32 / 128 / 128) * 128;
    assert!(Request::new(1, 1, 128, largest_rows, device()).is_ok());
    assert_eq!(
        Request::new(1, 1, 128, largest_rows + 128, device()),
        Err(Error::Extent)
    );
    assert!(Request::new(u32::MAX, u32::MAX, 128, u32::MAX - 127, device()).is_err());
}

#[test]
fn quantization_plan_and_all_six_output_spans_are_exact_and_non_aliasing() {
    let plan = plan();
    plan.validate().unwrap();
    let spans: Vec<_> = plan
        .bytes
        .all()
        .into_iter()
        .enumerate()
        .map(|(i, bytes)| Span {
            address: (i as u64 + 1) << 20,
            bytes,
        })
        .collect();
    let bindings = Bindings {
        query_input: spans[0],
        key_input: spans[1],
        value_input: spans[2],
        query: spans[3],
        key: spans[4],
        value: spans[5],
        query_scales: spans[6],
        key_scales: spans[7],
        value_scales: spans[8],
    };
    bindings.validate(&plan).unwrap();
    let mut overlap = bindings;
    overlap.value_scales = overlap.key_scales;
    assert_eq!(overlap.validate(&plan), Err(Error::Span));
    let mut overlap_input = bindings;
    overlap_input.query.address = bindings.key_input.address;
    assert_eq!(overlap_input.validate(&plan), Err(Error::Span));
    let mut short_input = bindings;
    short_input.query_input.bytes -= 2;
    assert_eq!(short_input.validate(&plan), Err(Error::Span));
    let mut unaligned_input = bindings;
    unaligned_input.value_input.address += 2;
    assert_eq!(unaligned_input.validate(&plan), Err(Error::Span));
    let mut forged = plan;
    forged.value_shared_bytes = 0;
    assert_eq!(forged.validate(), Err(Error::Plan));
    assert_ne!(forged.fingerprint_material(), plan.fingerprint_material());
}
