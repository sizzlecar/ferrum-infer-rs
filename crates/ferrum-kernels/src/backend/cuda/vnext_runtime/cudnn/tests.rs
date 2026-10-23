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

fn convolution_key() -> ConvolutionKey {
    ConvolutionKey {
        shape: CudnnConv2dDescriptor {
            input: [2, 4, 8, 9],
            filter: [6, 2, 3, 3],
            output: [2, 6, 8, 9],
            padding: [1, 1],
            stride: [1, 1],
            dilation: [1, 1],
            groups: 2,
        },
        workspace_cap: 128,
    }
}

// CPU-only callback for the same publication/quarantine boundary as the native
// call. It records selected math and the full supplied cap, not just required
// bytes. No CUDA context, library, allocation or timing result is fabricated.
struct MockConvolution {
    candidate: ForwardPerformance,
    required: usize,
    fail_at: Option<&'static str>,
    calls: Vec<&'static str>,
    forwards: Vec<(i32, i32, usize)>,
}
impl MockConvolution {
    fn new() -> Self {
        Self {
            candidate: ForwardPerformance {
                algo: 1,
                status: ffi::SUCCESS,
                determinism: ffi::DETERMINISTIC,
                memory: 32,
                math_type: ffi::TENSOR_OP_MATH,
                ..ForwardPerformance::default()
            },
            required: 64,
            fail_at: None,
            calls: Vec::new(),
            forwards: Vec::new(),
        }
    }

    fn call(&mut self, label: &'static str) -> Result<(), String> {
        self.calls.push(label);
        if self.fail_at == Some(label) {
            Err(label.into())
        } else {
            Ok(())
        }
    }

    fn count(&self, label: &str) -> usize {
        self.calls.iter().filter(|&&call| call == label).count()
    }

    fn run(
        &mut self,
        memo: &mut ConvolutionMemo,
        failed: &mut bool,
        key: ConvolutionKey,
    ) -> Result<(), String> {
        memo.invoke(failed, key, |cached| {
            self.call("configure")?;
            let selected = if let Some(selected) = cached {
                self.call("selected math")?;
                selected
            } else {
                self.call("algorithms")?;
                let candidate = choose_algorithm(&[self.candidate], key.workspace_cap)
                    .ok_or("no candidate fits policy")?;
                self.call("selected math")?;
                self.call("workspace")?;
                let selected = ConvolutionAlgorithm {
                    algo: candidate.algo,
                    math_type: candidate.math_type,
                    required_bytes: self.required,
                };
                selected.validate(key.workspace_cap)?;
                selected
            };
            self.call("forward")?;
            self.forwards
                .push((selected.algo, selected.math_type, key.workspace_cap));
            self.call("clear stream")?;
            Ok(selected)
        })
    }
}

#[test]
fn convolution_memo_reuses_exact_selection_across_shapes_but_not_owners() {
    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    let mut mock = MockConvolution::new();
    let a = convolution_key();
    let mut b = a;
    b.shape.input[2] += 1;
    mock.run(&mut memo, &mut failed, a).unwrap();
    mock.candidate.algo = 2;
    mock.candidate.math_type = 0;
    mock.required = 96;
    mock.run(&mut memo, &mut failed, b).unwrap();
    mock.run(&mut memo, &mut failed, a).unwrap();
    assert_eq!(mock.count("algorithms"), 2);
    assert_eq!(mock.count("workspace"), 2);
    assert_eq!(mock.count("configure"), 3);
    assert_eq!(mock.count("selected math"), 3);
    assert_eq!(mock.count("clear stream"), 3);
    assert_eq!(mock.forwards, [(1, 1, 128), (2, 0, 128), (1, 1, 128)]);
    assert_eq!(memo.lookup(a).unwrap().required_bytes, 64);
    assert_eq!(memo.lookup(b).unwrap().required_bytes, 96);
    assert!(!failed);

    let mut independent_owner = ConvolutionMemo::new();
    mock.run(&mut independent_owner, &mut failed, a).unwrap();
    assert_eq!(mock.count("algorithms"), 3);
    assert_eq!(independent_owner.lookup(a).unwrap().algo, 2);
    assert_eq!(memo.lookup(a).unwrap().algo, 1);
}

#[test]
fn convolution_memo_key_includes_every_descriptor_dimension_and_workspace_cap() {
    let key = convolution_key();
    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    let mut mock = MockConvolution::new();
    mock.run(&mut memo, &mut failed, key).unwrap();
    let mut variants = Vec::new();
    for array in 0..6 {
        let len = if array < 3 { 4 } else { 2 };
        for index in 0..len {
            let mut changed = key;
            let dimensions: &mut [i32] = match array {
                0 => &mut changed.shape.input,
                1 => &mut changed.shape.filter,
                2 => &mut changed.shape.output,
                3 => &mut changed.shape.padding,
                4 => &mut changed.shape.stride,
                _ => &mut changed.shape.dilation,
            };
            dimensions[index] += 1;
            variants.push(changed);
        }
    }
    let mut groups = key;
    groups.shape.groups += 1;
    variants.push(groups);
    variants.push(ConvolutionKey {
        workspace_cap: key.workspace_cap + 1,
        ..key
    });
    for changed in &variants {
        assert!(memo.lookup(*changed).is_none());
        mock.run(&mut memo, &mut failed, *changed).unwrap();
    }
    assert_eq!(mock.count("algorithms"), 1 + variants.len());
    mock.run(&mut memo, &mut failed, key).unwrap();
    assert_eq!(mock.count("algorithms"), 1 + variants.len());
}

#[test]
fn convolution_memo_full_capacity_preserves_uncached_execution_and_old_entries() {
    let key = convolution_key();
    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    let mut mock = MockConvolution::new();
    for index in 0..CONVOLUTION_MEMO_CAPACITY {
        let mut next = key;
        next.shape.input[0] += i32::try_from(index).unwrap();
        mock.run(&mut memo, &mut failed, next).unwrap();
    }
    let mut overflow = key;
    overflow.shape.input[0] += i32::try_from(CONVOLUTION_MEMO_CAPACITY).unwrap();
    for _ in 0..2 {
        mock.run(&mut memo, &mut failed, overflow).unwrap();
        assert!(memo.lookup(overflow).is_none());
    }
    assert_eq!(memo.len, CONVOLUTION_MEMO_CAPACITY);
    let queries = mock.count("algorithms");
    mock.run(&mut memo, &mut failed, key).unwrap();
    assert_eq!(mock.count("algorithms"), queries);
    assert_eq!(queries, CONVOLUTION_MEMO_CAPACITY + 2);
    assert!(!failed);
}

#[test]
fn convolution_memo_failures_and_unwind_never_publish_or_clear_quarantine() {
    let key = convolution_key();
    for stage in [
        "configure",
        "algorithms",
        "selected math",
        "workspace",
        "forward",
        "clear stream",
    ] {
        let mut memo = ConvolutionMemo::new();
        let mut failed = false;
        let mut mock = MockConvolution::new();
        mock.fail_at = Some(stage);
        assert!(mock.run(&mut memo, &mut failed, key).is_err());
        assert!(failed, "{stage}");
        assert_eq!(memo.len, 0, "{stage}");
        let calls = mock.calls.len();
        mock.fail_at = None;
        assert!(mock.run(&mut memo, &mut failed, key).is_err());
        assert_eq!(mock.calls.len(), calls);
    }
    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        memo.invoke(&mut failed, key, |_| panic!("native callback unwind"))
    }));
    assert!(result.is_err());
    assert!(failed);
    assert_eq!(memo.len, 0);

    for stage in ["forward", "clear stream"] {
        let mut memo = ConvolutionMemo::new();
        let mut failed = false;
        let mut mock = MockConvolution::new();
        mock.run(&mut memo, &mut failed, key).unwrap();
        let original = memo.lookup(key);
        mock.fail_at = Some(stage);
        assert!(mock.run(&mut memo, &mut failed, key).is_err());
        assert!(failed);
        assert_eq!(memo.lookup(key), original);
        assert_eq!(mock.count("algorithms"), 1);
    }
}

#[test]
fn convolution_memo_rejects_policy_and_actual_workspace_failures_before_forward() {
    let key = convolution_key();
    let valid = MockConvolution::new().candidate;
    for candidate in [
        ForwardPerformance { status: 1, ..valid },
        ForwardPerformance {
            determinism: 0,
            ..valid
        },
        ForwardPerformance {
            memory: key.workspace_cap + 1,
            ..valid
        },
        ForwardPerformance {
            math_type: 2,
            ..valid
        },
        ForwardPerformance { algo: -1, ..valid },
        ForwardPerformance {
            algo: ffi::ALGORITHMS as i32,
            ..valid
        },
    ] {
        let mut memo = ConvolutionMemo::new();
        let mut failed = false;
        let mut mock = MockConvolution::new();
        mock.candidate = candidate;
        assert!(mock.run(&mut memo, &mut failed, key).is_err());
        assert!(failed);
        assert!(mock.forwards.is_empty());
        assert_eq!(memo.len, 0);
    }
    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    let mut mock = MockConvolution::new();
    // Heuristic memory fits, but the selected-math workspace query does not.
    mock.required = key.workspace_cap + 1;
    assert!(mock.run(&mut memo, &mut failed, key).is_err());
    assert!(failed);
    assert_eq!(mock.count("workspace"), 1);
    assert!(mock.forwards.is_empty());
    assert_eq!(memo.len, 0);

    let mut memo = ConvolutionMemo::new();
    let mut failed = false;
    mock.required = 0;
    mock.candidate.memory = 0;
    let zero_workspace = ConvolutionKey {
        workspace_cap: 0,
        ..key
    };
    mock.run(&mut memo, &mut failed, zero_workspace).unwrap();
    mock.run(&mut memo, &mut failed, zero_workspace).unwrap();
    assert_eq!(mock.forwards, [(1, 1, 0), (1, 1, 0)]);
}
