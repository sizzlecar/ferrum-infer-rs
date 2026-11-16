use super::*;
pub(super) fn precompute(
    a: &Config,
    context: &Arc<runtime::Context>,
    package: &Arc<FastH3TransformerPackage>,
    schedule: &FastH3ConditioningSchedule,
    host: &HostMemoryAccount,
) -> Result<Vec<FastH3ConditioningTable>, Failure> {
    let bounds = FastH3TimeConditioningLimits {
        maximum_time_rows: schedule.timesteps().len() as u64,
        maximum_metadata_bytes: a.maximum_metadata_bytes,
        maximum_stored_bytes: a.device_bytes,
    };
    let mut tables = Vec::with_capacity(package.config().layers());
    for block in 0..package.config().layers() {
        let c = FastH3TimeConditioningConfig::from_package(
            package,
            block,
            schedule.timesteps().len() as u64,
            bounds,
        )
        .map_err(|e| error("Time configuration", e))?;
        let source = FastH3TimeConditioningSource::new(package, &c, host, bounds)
            .map_err(|e| error("Time source", e))?;
        let family = TypedFamilyRegistration::new(
            FastH3TimeConditioningFamily::new(bounds).map_err(|e| error("Time family", e))?,
        )
        .prepare_with_profile(
            &serde_json::to_value(&c).map_err(|e| error("Time config", e))?,
            &NumericalProfileId::new(FastH3TimeConditioningFamily::PROFILE_ID)
                .map_err(|e| error("Time profile", e))?,
        )
        .map_err(|e| error("Time definition", e))?;
        let compiled = Arc::new(
            CompiledTensorComponent::compile(
                family,
                &context.registry,
                &context.catalog,
                &context.policy,
                &context.materializers,
                &c.compile_options().map_err(|e| error("Time options", e))?,
            )
            .map_err(|e| Failure::retain("Time compile", e, context.clone()))?,
        );
        let input = c
            .managed_schedule_features(host, schedule)
            .map_err(|e| error("Time features", e))?;
        let phase = Arc::new((
            context.clone(),
            compiled.clone(),
            package.clone(),
            input.clone(),
        ));
        let mut session = compiled
            .initialize_retained(
                context.runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&source),
                initialization(a, &format!("time{block}"))
                    .map_err(|e| error("Time static policy", e))?,
                phase.clone(),
            )
            .map_err(|e| Failure::retain("Time initialize", e, phase.clone()))?;
        let outputs = match session.session_mut().execute(
            id(format!("run.dit.time{block}")),
            id(format!("request.dit.time{block}")),
            vec![input],
        ) {
            Ok(outputs) => outputs,
            Err(e) => return Err(Failure::retain("Time execute", e, (session, phase))),
        };
        let (returned, _) = session
            .close()
            .map_err(|e| Failure::retain("Time close", e, phase.clone()))?;
        drop(returned);
        drop(phase);
        tables.push(
            FastH3ConditioningTable::from_execution(c, schedule, outputs, bounds)
                .map_err(|e| error("Time output", e))?,
        );
        eprintln!(
            "{}",
            json!({"stage":"time_precompute","block":block,"retired":true})
        );
    }
    Ok(tables)
}
