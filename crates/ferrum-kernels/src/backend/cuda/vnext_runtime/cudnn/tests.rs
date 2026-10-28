use super::*;

#[test]
fn algorithm_selection_filters_capacity_determinism_status_and_math() {
    let valid = ForwardPerformance {
        algo: 1,
        status: 0,
        time: 1.0,
        memory: 64,
        determinism: ffi::DETERMINISTIC,
        math_type: ffi::TENSOR_OP_MATH,
        reserved: [0; 3],
    };
    let candidates = [
        ForwardPerformance { status: 2, ..valid },
        ForwardPerformance {
            determinism: 0,
            ..valid
        },
        ForwardPerformance {
            memory: 65,
            ..valid
        },
        ForwardPerformance {
            math_type: 2,
            ..valid
        },
        ForwardPerformance { algo: 8, ..valid },
        valid,
    ];
    assert_eq!(choose_algorithm(&candidates, 64).unwrap().algo, 1);
    assert!(choose_algorithm(&candidates, 63).is_none());
    assert!(choose_algorithm(&[], 64).is_none());
    assert_eq!(
        choose_algorithm(&[ForwardPerformance { memory: 0, ..valid }], 0)
            .unwrap()
            .memory,
        0
    );
}

#[test]
fn config_rejects_implicit_paths_and_unbounded_or_unlabelled_allowances() {
    let config = CudaCudnnConfig {
        library_path: PathBuf::from("/deployment/libcudnn.so.9"),
        dependency_libraries: Vec::new(),
        opaque_memory_bytes: 64,
        workspace_limit_bytes: 0,
        opaque_memory_profile: "experimental-test-v1".into(),
    };
    assert!(config.validate().is_ok());
    assert!(CudaCudnnConfig {
        library_path: PathBuf::from("libcudnn.so.9"),
        ..config.clone()
    }
    .validate()
    .is_err());
    assert!(CudaCudnnConfig {
        opaque_memory_bytes: 0,
        ..config.clone()
    }
    .validate()
    .is_err());
    assert!(CudaCudnnConfig {
        workspace_limit_bytes: isize::MAX as u64 + 1,
        ..config.clone()
    }
    .validate()
    .is_err());
    assert!(CudaCudnnConfig {
        opaque_memory_profile: " ".into(),
        ..config
    }
    .validate()
    .is_err());
}

#[test]
fn forward_performance_abi_matches_64_bit_cudnn_header() {
    if std::mem::size_of::<usize>() == 8 {
        assert_eq!(std::mem::size_of::<ForwardPerformance>(), 48);
        assert_eq!(std::mem::offset_of!(ForwardPerformance, memory), 16);
        assert_eq!(std::mem::offset_of!(ForwardPerformance, determinism), 24);
        assert_eq!(std::mem::offset_of!(ForwardPerformance, reserved), 32);
    }
}

fn convolution_shape() -> CudnnConv2dDescriptor {
    CudnnConv2dDescriptor {
        input: [2, 4, 8, 9],
        filter: [6, 2, 3, 3],
        output: [2, 6, 8, 9],
        padding: [1, 1],
        stride: [1, 1],
        dilation: [1, 1],
        groups: 2,
    }
}

fn selected_algorithm() -> ConvolutionAlgorithm {
    ConvolutionAlgorithm {
        algo: 1,
        math_type: ffi::TENSOR_OP_MATH,
        required_bytes: 64,
    }
}

#[test]
fn convolution_uses_actual_workspace_and_rejects_cap_sized_or_invalid_bindings() {
    let selected = selected_algorithm();
    assert!(selected.validate(128).is_ok());
    assert!(selected.validate(63).is_err());
    assert!(selected.validate_workspace(256, 64).is_ok());
    assert!(selected.validate_workspace(256, 128).is_err());
    assert!(selected.validate_workspace(0, 64).is_err());
    assert!(selected.validate_workspace(256, 0).is_err());
    let zero = ConvolutionAlgorithm {
        required_bytes: 0,
        ..selected
    };
    assert!(zero.validate(128).is_ok());
    assert!(zero.validate_workspace(0, 0).is_ok());
    assert!(zero.validate_workspace(256, 0).is_err());
}

#[test]
fn prepared_convolution_binds_owner_shape_policy_and_actual_selection() {
    // Metadata-only owners exercise encode validation without a CUDA context,
    // library or native allocation. Native execution is never invoked.
    let make_owner = || {
        Arc::new(CudaCudnnOwner {
            inner: Mutex::new(None),
            fingerprint: "test-owner".into(),
            workspace_limit_bytes: 128,
        })
    };
    let owner = make_owner();
    let shape = convolution_shape();
    let selected = selected_algorithm();
    let fingerprint = convolution_fingerprint(owner.fingerprint(), shape, 128, selected);
    let prepared = PreparedConv {
        owner: Arc::clone(&owner),
        shape,
        selected,
        fingerprint: fingerprint.clone(),
    };
    assert_eq!(prepared.workspace_bytes(), 64);
    assert!(prepared
        .validate_invocation(&owner, shape, Some(&fingerprint))
        .is_ok());
    assert!(prepared
        .validate_invocation(&make_owner(), shape, Some(&fingerprint))
        .is_err());
    assert!(prepared.validate_invocation(&owner, shape, None).is_err());
    assert!(prepared
        .validate_invocation(&owner, shape, Some("another-selection"))
        .is_err());

    let mut variants = Vec::new();
    for array in 0..6 {
        let len = if array < 3 { 4 } else { 2 };
        for index in 0..len {
            let mut changed = shape;
            let dimensions: &mut [i32] = match array {
                0 => &mut changed.input,
                1 => &mut changed.filter,
                2 => &mut changed.output,
                3 => &mut changed.padding,
                4 => &mut changed.stride,
                _ => &mut changed.dilation,
            };
            dimensions[index] += 1;
            variants.push(changed);
        }
    }
    variants.push(CudnnConv2dDescriptor { groups: 1, ..shape });
    for changed in variants {
        assert!(prepared
            .validate_invocation(&owner, changed, Some(&fingerprint))
            .is_err());
        assert_ne!(
            convolution_fingerprint(owner.fingerprint(), changed, 128, selected),
            fingerprint
        );
    }
    for changed in [
        ConvolutionAlgorithm {
            algo: 2,
            ..selected
        },
        ConvolutionAlgorithm {
            math_type: 0,
            ..selected
        },
        ConvolutionAlgorithm {
            required_bytes: 96,
            ..selected
        },
    ] {
        assert_ne!(
            convolution_fingerprint(owner.fingerprint(), shape, 128, changed),
            fingerprint
        );
    }
    assert_ne!(
        convolution_fingerprint(owner.fingerprint(), shape, 256, selected),
        fingerprint
    );
    assert_ne!(
        convolution_fingerprint("another-owner", shape, 128, selected),
        fingerprint
    );
}

#[test]
fn prepared_policy_rejection_keeps_owner_reusable_but_query_failure_quarantines() {
    let mut failed = false;
    assert!(prepare_convolution_selection(&mut failed, 128, || Ok(None)).is_err());
    assert!(!failed);
    assert!(
        prepare_convolution_selection(&mut failed, 32, || { Ok(Some(selected_algorithm())) })
            .is_err()
    );
    assert!(!failed);
    assert_eq!(
        prepare_convolution_selection(&mut failed, 128, || Ok(Some(selected_algorithm()))).unwrap(),
        selected_algorithm()
    );
    assert!(!failed);
    assert!(
        prepare_convolution_selection(&mut failed, 128, || Err("query failed".into())).is_err()
    );
    assert!(failed);
    let mut retried = false;
    assert!(prepare_convolution_selection(&mut failed, 128, || {
        retried = true;
        Ok(Some(selected_algorithm()))
    })
    .is_err());
    assert!(!retried);
}

#[test]
fn native_convolution_errors_and_unwinds_quarantine_without_retry() {
    let mut failed = false;
    assert_eq!(guarded_convolution(&mut failed, || Ok(7)).unwrap(), 7);
    assert!(!failed);
    assert!(guarded_convolution::<()>(&mut failed, || Err("native failure".into())).is_err());
    assert!(failed);
    let mut retried = false;
    assert!(guarded_convolution(&mut failed, || {
        retried = true;
        Ok(())
    })
    .is_err());
    assert!(!retried);

    let mut failed = false;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: Result<(), String> = guarded_convolution(&mut failed, || panic!("native unwind"));
    }));
    assert!(result.is_err());
    assert!(failed);
}
