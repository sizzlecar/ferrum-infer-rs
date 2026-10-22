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
