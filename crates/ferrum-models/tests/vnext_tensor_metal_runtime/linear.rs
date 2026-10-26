use super::*;

pub(super) fn verify(
    runtime: &Arc<MetalDeviceRuntime>,
    registry: &OperationRuntimeRegistry<MetalDeviceRuntime>,
    catalog: &CapabilityCatalog,
    policy: &TensorPolicy,
    host: &HostMemoryAccount,
) {
    let kind = MetalTensorOperation::DenseLinear;
    // Exact tiles, every axis with a tail, singleton GEMV and a longer K.
    // K4096 covers the MPP64 specialization alone and together with the
    // unchanged MPP32 remainder and SIMD bottom/right strips.
    for (case, (m, n, k)) in [
        (1, 1, 1),
        (31, 63, 31),
        (32, 64, 32),
        (32, 64, 1),
        (64, 128, 256),
        (64, 64, 4096),
        (97, 129, 4096),
        (64, 192, 129),
        (96, 192, 129),
        (97, 193, 129),
        (65, 129, 129),
        (33, 65, 33),
        (65, 17, 129),
        (17, 128, 1024),
    ]
    .into_iter()
    .enumerate()
    {
        let mut config = Config::new(kind, m, k, false);
        config.outputs = Some(n);
        let prepared = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(&config).unwrap(), &id(kind.profile()))
            .unwrap();
        let mut options = ProgramPlanCompileOptions::new(config.input_specs()).unwrap();
        options.retain_completion_value(id(OUTPUT));
        let compiled = ProgramPlanCompiler::compile(
            &prepared,
            catalog,
            policy,
            &registry.planning(),
            &options,
        )
        .unwrap();
        let executable = compiled.executable();
        let resources = match executable
            .execution_plan()
            .provision_static(
                runtime.clone(),
                id(format!("request.metal.linear.provision.{case}")),
            )
            .unwrap()
            .into_provisioning()
        {
            StaticProvisioning::NoStatic(provisioned) => provisioned.into_plan_runtime(),
            _ => panic!("dynamic matrix fixture has no stored weights"),
        };
        let mut executor = PreparedTensorExecutor::new(
            executable.clone(),
            registry.bind_plan(executable).unwrap(),
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .unwrap();
        let values: Vec<Vec<f32>> = [m, n]
            .into_iter()
            .enumerate()
            .map(|(port, rows)| {
                (0..rows)
                    .flat_map(|row| {
                        (0..k).map(move |column| {
                            let exponent = match column % 4 {
                                0 => 80,
                                1 => -80,
                                _ => 0,
                            };
                            let scale = 2_f32.powi(if port == 0 { exponent } else { -exponent });
                            // BF16 exponent range exceeds FP16. Corresponding products
                            // remain finite and normal, so an FP16 detour is observable.
                            rounds(
                                (((row * 7 + column * 17 + port as u64 * 13) % 43) as f32 - 21.0)
                                    / 8.0
                                    * scale,
                            )
                        })
                    })
                    .collect()
            })
            .collect();
        let mut previous = None;
        for repeat in 0..2 {
            let output = executor
                .execute(
                    id(format!("run.metal.linear.{case}.{repeat}")),
                    id(format!("request.metal.linear.{case}.{repeat}")),
                    upload(host, &config, &values),
                )
                .unwrap();
            let bytes = output[0].bytes().bytes();
            assert_eq!(bytes.len() as u64, m * n * 2);
            for (index, word) in bytes.chunks_exact(2).enumerate() {
                let row = index / n as usize;
                let column = index % n as usize;
                let mut sum = 0_f64;
                let mut absolute = 0_f64;
                for inner in 0..k as usize {
                    let product = f64::from(values[0][row * k as usize + inner])
                        * f64::from(values[1][column * k as usize + inner]);
                    sum += product;
                    absolute += product.abs();
                }
                let rounding = (2 * k) as f64 * 2_f64.powi(-24);
                let accumulation = rounding / (1.0 - rounding) * absolute;
                let oracle_rounding = (2 * k) as f64 * f64::EPSILON;
                let oracle = oracle_rounding / (1.0 - oracle_rounding) * absolute;
                let bound =
                    accumulation + (sum.abs() + accumulation) / 256.0 + 2_f64.powi(-134) + oracle;
                let actual = f64::from(
                    bf16::from_bits(u16::from_le_bytes(word.try_into().unwrap())).to_f32(),
                );
                assert!(actual.is_finite() && (actual - sum).abs() <= bound,
                    "M={m} N={n} K={k} [{row},{column}]: {actual} versus {sum}, operation bound {bound}");
            }
            if let Some(previous) = &previous {
                assert_eq!(bytes, previous);
            }
            previous = Some(bytes.to_vec());
        }
        drop((executor, compiled));
        assert!(matches!(
            PlanRuntimeResources::close(resources),
            Ok(PlanRuntimeCloseOutcome::Closed(_))
        ));
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        eprintln!(
            "Metal BF16 linear M={m} N={n} K={k}: FP64 oracle, repeat and retirement verified"
        );
    }
}
