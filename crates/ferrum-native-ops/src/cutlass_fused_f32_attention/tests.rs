use super::*;

#[test]
fn checked_geometry_and_actual_span_boundaries_fail_closed() {
    for (b, s, h, d) in [(1, 4096, 16, 72), (4, 261, 32, 64)] {
        let request = CutlassFusedF32AttentionRequestV1::new(b, s, h, d, 0.125).unwrap();
        let elements = request.elements().unwrap();
        let stride = elements * 4 + 256;
        let span = |i| CutlassFusedF32AttentionSpanV1 {
            address: 256 + i * stride,
            elements,
        };
        let bindings = CutlassFusedF32AttentionBindingsV1 {
            query: span(0),
            key: span(1),
            value: span(2),
            output: span(3),
        };
        bindings.validate(&request).unwrap();
        for output in [
            bindings.query,
            bindings.key,
            bindings.value,
            CutlassFusedF32AttentionSpanV1 {
                address: bindings.query.address + 16,
                elements,
            },
            CutlassFusedF32AttentionSpanV1 {
                address: u64::MAX - 15,
                elements,
            },
            CutlassFusedF32AttentionSpanV1 {
                address: bindings.output.address + 4,
                elements,
            },
            CutlassFusedF32AttentionSpanV1 {
                address: bindings.output.address,
                elements: elements - 1,
            },
        ] {
            assert!(CutlassFusedF32AttentionBindingsV1 { output, ..bindings }
                .validate(&request)
                .is_err());
        }
        let mut invalid = request;
        invalid.reserved = 1;
        assert!(bindings.validate(&invalid).is_err());
    }
    for scale in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(CutlassFusedF32AttentionRequestV1::new(1, 1, 1, 64, scale).is_err());
    }
    for geometry in [
        (0, 1, 1, 64),
        (1, 0, 1, 64),
        (1, 1, 0, 64),
        (1, 1, 1, 65),
        (65536, 1, 1, 64),
        (1, 65535, 65535, 72),
    ] {
        assert!(CutlassFusedF32AttentionRequestV1::new(
            geometry.0, geometry.1, geometry.2, geometry.3, 1.0
        )
        .is_err());
    }
}

#[test]
fn native_info_requires_matching_tile_and_no_undeclared_workspace() {
    let request = CutlassFusedF32AttentionRequestV1::new(1, 4096, 16, 72, 0.125).unwrap();
    let info = CutlassFusedF32AttentionInfoV1 {
        abi: 1,
        size: 32,
        shared_bytes: 86016,
        threads: 256,
        query_tile: 64,
        key_tile: 128,
        global_workspace_bytes: 0,
    };
    info.validate_for(&request).unwrap();
    assert!(info
        .validate_for(&CutlassFusedF32AttentionRequestV1::new(1, 261, 32, 64, 0.125).unwrap())
        .is_err());
    assert!(CutlassFusedF32AttentionInfoV1 {
        global_workspace_bytes: 1,
        ..info
    }
    .validate_for(&request)
    .is_err());
    assert!(CutlassFusedF32AttentionInfoV1 {
        shared_bytes: 0,
        ..info
    }
    .validate_for(&request)
    .is_err());
    assert_ne!(
        request.fingerprint_material(),
        CutlassFusedF32AttentionRequestV1 {
            scale: 0.25,
            ..request
        }
        .fingerprint_material()
    );
    assert_ne!(
        info.fingerprint_material(),
        CutlassFusedF32AttentionInfoV1 {
            shared_bytes: 1,
            ..info
        }
        .fingerprint_material()
    );
}

#[test]
fn native_unit_needs_its_own_complete_abi_exports() {
    let unit = crate::CudaNativeBuildUnit::CutlassFusedF32Attention;
    assert_eq!(
        unit.artifact_operator(),
        CUTLASS_FUSED_F32_ATTENTION_OPERATOR
    );
    let exports: Vec<_> = CUTLASS_FUSED_F32_ATTENTION_EXPORTS
        .iter()
        .map(|x| (*x).to_owned())
        .collect();
    unit.validate_package_contract("1", &exports).unwrap();
    assert!(unit.validate_package_contract("2", &exports).is_err());
    assert!(unit.validate_package_contract("1", &exports[..1]).is_err());
    assert!(unit
        .validate_package_contract(
            "1",
            &[
                "ferrum_qualify_fused_f32_prepare".into(),
                "ferrum_qualify_fused_f32_run".into()
            ]
        )
        .is_err());
}
