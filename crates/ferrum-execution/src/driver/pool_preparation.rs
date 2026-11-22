//! Physical capacity preparation shares the compiled demand formulas and the
//! ordinary pool maintenance authority; it owns no allocator or request lease.

use super::*;

/// Bytes needed to run another verified shape while the optional backing
/// stays resident. Free bytes in one pool cannot serve a different pool.
fn capacity_with_prepared_pools<Pool: Ord>(
    requested_bytes: u64,
    prepared: &BTreeMap<Pool, u64>,
    other_workload: &BTreeMap<Pool, u64>,
) -> u128 {
    other_workload
        .iter()
        .fold(u128::from(requested_bytes), |total, (pool, &needed)| {
            let available = prepared.get(pool).copied().unwrap_or(0);
            total + u128::from(needed.saturating_sub(available))
        })
}

pub(super) fn prepare<R: DeviceRuntime>(
    resources: &Arc<PlanRuntimeResources<R>>,
    memory: &MemoryPlan,
    mode: ferrum_types::RuntimePoolPreparation,
    context_tokens: usize,
    step_tokens: u64,
    startup: Option<&ferrum_types::StartupMemoryPlan>,
) -> Result<()> {
    if mode == ferrum_types::RuntimePoolPreparation::OnDemand {
        return Ok(());
    }
    let context = u64::try_from(context_tokens)
        .map_err(|_| FerrumError::config("pool preparation context exceeds u64"))?;
    let resident = memory.maximum_active_sequences();
    let pool_bytes = |frontier, residents, wave, tokens| {
        memory
            .startup_resident_workload_pool_bytes(context, frontier, residents, wave, tokens)
            .map_err(|error| FerrumError::config(format!("compiled pool preparation: {error}")))
    };
    let (targets, other_workload) = match startup.and_then(|plan| {
        plan.serving_concurrency_search
            .as_ref()
            .map(|search| (plan, search))
    }) {
        Some((plan, search)) => {
            let wave = u32::try_from(search.decode_wave_sequences)
                .map_err(|_| FerrumError::config("startup decode wave exceeds u32"))?;
            // The search verified these workloads separately. Only the chosen
            // shape is allocated; the other must still fit beside idle backing.
            let decode = pool_bytes(1, resident, wave, u64::from(wave))?;
            let prefill = pool_bytes(context, 1, 1, step_tokens.min(context))?;
            if plan.decode_peak_bytes >= plan.context_peak_bytes {
                (decode, prefill)
            } else {
                (prefill, decode)
            }
        }
        None => (
            pool_bytes(context, resident, resident, step_tokens)?,
            BTreeMap::new(),
        ),
    };
    let requested_total = targets
        .values()
        .try_fold(memory.static_bytes(), |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| FerrumError::config("pool preparation bytes overflow u64"))
        })?;
    // The configured admission limits are not a promise that every live
    // sequence can hold its maximum context simultaneously. An oversized
    // preparation target must leave the original on-demand service runnable.
    let required_capacity =
        capacity_with_prepared_pools(requested_total, &targets, &other_workload);
    if required_capacity > u128::from(memory.usable_capacity_bytes()) {
        tracing::info!(target: "ferrum.runtime.pool_preparation",
            outcome = "on_demand", requested_total_bytes = requested_total,
            required_capacity_bytes = %required_capacity,
            budget_bytes = memory.usable_capacity_bytes(),
            "compiled pool preparation exceeds budget; retaining on-demand growth");
        return Ok(());
    }
    let phase = StartupPhaseTimer::start("dynamic_pool_preparation");
    match resources.ensure_pool_residency(targets) {
        Ok(receipt) => {
            tracing::info!(target: "ferrum.runtime.pool_preparation",
                outcome = "prepared", requested_total_bytes = requested_total,
                required_capacity_bytes = %required_capacity,
                growths = ?receipt.growths(),
                "compiled physical pool capacity prepared before readiness");
        }
        Err(
            error @ (VNextError::DeviceCapacityUnavailable(_)
            | VNextError::DynamicPoolResidentUnavailable(_)),
        ) => {
            // These typed denials precede native allocation. Contention is not
            // permission to evict owners or reject otherwise valid startup.
            tracing::info!(target: "ferrum.runtime.pool_preparation",
                outcome = "on_demand", requested_total_bytes = requested_total,
                required_capacity_bytes = %required_capacity,
                reason = %error,
                "optional pool preparation denied; retaining on-demand growth");
        }
        Err(error) => {
            return Err(FerrumError::model(format!(
                "runtime pool preparation failed: {error}"
            )))
        }
    }
    phase.finish();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_backing_must_leave_room_for_the_other_verified_workload() {
        let shallow = BTreeMap::from([("state", 32), ("kv", 16), ("workspace", 16)]);
        let long = BTreeMap::from([("state", 8), ("kv", 32), ("workspace", 48)]);
        // 16 static bytes are shared. Each shape alone fits 112, even though
        // the shallow shape's idle state cannot fund the longer prefill.
        assert_eq!(capacity_with_prepared_pools(80, &shallow, &long), 128);
        assert_eq!(capacity_with_prepared_pools(104, &long, &shallow), 128);
        assert_eq!(capacity_with_prepared_pools(80, &shallow, &shallow), 80);
        let additional_pool = BTreeMap::from([("separate", 16)]);
        assert_eq!(
            capacity_with_prepared_pools(u64::MAX, &shallow, &additional_pool),
            u128::from(u64::MAX) + 16
        );
    }
}
