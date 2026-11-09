//! On-demand continuation replay for the declared tail-two causal ABI. Each
//! initialized program owns one stable lane; frames and selected settlement
//! still belong to Core. Prefill never captures user state for validation.
use super::*;

#[cfg(feature = "test-support")]
pub(super) mod diagnostic;

const TWO_ROW_CLASS: &str = "ferrum.causal.tail-two.continuation";

#[cfg(test)]
mod tests;

pub(super) fn resolve_policy(
    policy: ResolvedRuntimePolicy,
    executable: bool,
) -> Result<ResolvedRuntimePolicy> {
    resolve_policy_inner(policy, executable, false)
}

pub(super) fn resolve_policy_with_eager_boundaries(
    policy: ResolvedRuntimePolicy,
    executable: bool,
) -> Result<ResolvedRuntimePolicy> {
    resolve_policy_inner(policy, executable, true)
}

fn resolve_policy_inner(
    policy: ResolvedRuntimePolicy,
    executable: bool,
    declared_eager_boundaries: bool,
) -> Result<ResolvedRuntimePolicy> {
    let Some(reusable) = policy.reusable_execution() else {
        return Ok(policy);
    };
    let Some(programs) = reusable.program_policy() else {
        return Ok(policy);
    };
    if !executable
        || programs.preparation() != ReusableExecutionProgramPreparation::OnDemand
        || reusable.maximum_reusable_lanes() != 1
    {
        return Err(FerrumError::unsupported(
            "bounded causal replay requires one on-demand lane on an executable program",
        ));
    }
    let two = ReusableExecutionClassId::new(TWO_ROW_CLASS).map_err(model_error)?;
    let mut buckets = reusable.buckets().to_vec();
    let mut shapes = programs.programs().to_vec();
    for spec in programs.programs() {
        let ReusableExecutionProgramShape::UniformDecode {
            request_capacity,
            query_tokens_per_sequence: 1,
            ..
        } = spec.shape()
        else {
            continue;
        };
        let tokens = u64::from(request_capacity) * 2;
        if tokens > policy.admission().maximum_scheduled_tokens {
            continue;
        }
        buckets.push(
            ReusableExecutionBucketSpec::new(
                two.clone(),
                ReusableExecutionCapacity::new(request_capacity, tokens, 1).map_err(model_error)?,
            )
            .map_err(model_error)?,
        );
        shapes.push(
            ReusableExecutionProgramSpec::new(
                two.clone(),
                ReusableExecutionProgramShape::uniform_decode(request_capacity, 2)
                    .map_err(model_error)?,
            )
            .map_err(model_error)?,
        );
    }
    let programs = ReusableExecutionProgramPolicy::exact_on_demand(shapes)
        .and_then(|p| p.with_on_demand_admission(programs.on_demand_admission()))
        .map_err(model_error)?;
    let mut resolved = ReusableExecutionPolicy::new(1, buckets)
        .and_then(|p| p.with_program_policy(programs))
        .map_err(model_error)?;
    if let Some(maximum) = reusable.maximum_executables_per_lane() {
        resolved = resolved
            .with_maximum_executables_per_lane(maximum)
            .map_err(model_error)?;
    }
    ResolvedRuntimePolicy::new(
        policy.policy_id(),
        policy.version(),
        policy.scheduling(),
        policy.memory().clone(),
        policy.admission().clone(),
        policy.attention_execution(),
        if declared_eager_boundaries {
            // This permits selecting eager-only providers, not replaying
            // them. Dispatch separately rejects ineligible Static/Dynamic
            // topology and enforces every declared eager boundary.
            ExecutionDeterminismRequirement::BitwiseSameRuntime
        } else {
            policy.execution_determinism()
        },
        Some(resolved),
    )
    .map_err(model_error)
}

fn model_error(error: impl fmt::Display) -> FerrumError {
    FerrumError::model(format!("causal replay: {error}"))
}

struct StableLane<R: DeviceRuntime> {
    lane: Arc<ExecutionLane<R>>,
    ready: bool,
}

pub(super) struct CausalReplay<R: DeviceRuntime> {
    preparation: DeviceReusableExecutionPlan,
    lane: Mutex<Option<StableLane<R>>>,
    catalog: RwLock<Option<Arc<VNextReusableExecutionCatalog>>>,
    pub(super) direct_waves: AtomicU64,
    pub(super) fallback_waves: AtomicU64,
    #[cfg(feature = "test-support")]
    pub(super) diagnostic_enabled: AtomicBool,
}

impl<R: DeviceRuntime> CausalReplay<R> {
    pub(super) fn from_plan(plan: &ResolvedModelPlan) -> Result<Option<Self>> {
        let Some(memory) = plan
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
        else {
            return Ok(None);
        };
        let Some(programs) = memory.program_policy() else {
            return Ok(None);
        };
        if memory.maximum_reusable_lanes() != 1
            || programs.preparation() != ReusableExecutionProgramPreparation::OnDemand
        {
            return Err(model_error(
                "resolved plan has unsupported lane/preparation policy",
            ));
        }
        let maximum = usize::try_from(
            memory
                .capped_device_executables_per_lane()
                .unwrap_or_else(|| memory.maximum_device_executables()),
        )
        .map_err(model_error)?;
        let preparation = DeviceReusableExecutionPlan::on_demand(maximum)
            .and_then(|p| p.with_on_demand_admission(programs.on_demand_admission()))
            .map_err(model_error)?;
        Ok(Some(Self {
            preparation,
            lane: Mutex::new(None),
            catalog: RwLock::new(None),
            direct_waves: AtomicU64::new(0),
            fallback_waves: AtomicU64::new(0),
            #[cfg(feature = "test-support")]
            diagnostic_enabled: AtomicBool::new(false),
        }))
    }

    fn lane(&self, resources: &Arc<PlanRuntimeResources<R>>) -> Result<Arc<ExecutionLane<R>>> {
        let mut owned = self.lane.lock();
        if owned.is_none() {
            let lane = resources.create_execution_lane().map_err(model_error)?;
            // Retain a failed native owner too. A failed configuration cannot
            // silently obtain a fresh lane on the next call.
            *owned = Some(StableLane { lane, ready: false });
            let entry = owned.as_mut().unwrap();
            let receipt = entry
                .lane
                .configure_reusable_executables(self.preparation)
                .map_err(model_error)?;
            if receipt.state() != DeviceReusableExecutionPreparationState::Ready
                || receipt.maximum_executables() != self.preparation.maximum_executables() as u64
                || receipt.resident_executables() != 0
                || receipt.captured_executables() != 0
            {
                return Err(model_error("on-demand lane did not start empty and ready"));
            }
            entry.ready = true;
        }
        let entry = owned.as_ref().unwrap();
        if !entry.ready {
            return Err(model_error("original replay lane preparation failed"));
        }
        Ok(Arc::clone(&entry.lane))
    }

    pub(super) fn program(
        &self,
        session: &CausalProgramSession<R>,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<Option<DeviceReusableExecutionProgram>> {
        if !self
            .lane
            .lock()
            .as_ref()
            .is_some_and(|owned| owned.ready && Arc::ptr_eq(&owned.lane, lane))
        {
            return Err(model_error("causal wave belongs to another execution lane"));
        }
        let Some(id) = OperationDispatch::reusable_execution_program_id_for_wave(
            session.providers.providers(),
            &session.plan,
            wave,
            lane,
        )
        .map_err(model_error)?
        else {
            return Ok(None);
        };
        Ok(self
            .catalog
            .read()
            .as_ref()
            .filter(|catalog| catalog.lane_epoch == lane.reusable_execution_epoch())
            .and_then(|catalog| catalog.programs.get(&id))
            .filter(|program| program.has_resident_segments())
            .cloned())
    }

    pub(super) fn refresh(
        &self,
        session: &CausalProgramSession<R>,
        lane: &Arc<ExecutionLane<R>>,
    ) -> Result<()> {
        let Some(snapshot) = lane.try_reusable_execution_catalog().map_err(model_error)? else {
            // Another actual owner may hold the stream; an old epoch will miss
            // safely and a later ordinary submission can publish a new catalog.
            return Ok(());
        };
        let (lane_epoch, programs) = snapshot.into_parts();
        if programs.len() > self.preparation.maximum_executables() {
            return Err(model_error("causal catalog exceeds its declared capacity"));
        }
        let mut catalog = BTreeMap::new();
        for program in programs {
            let id = program.program_id();
            if id.plan_hash() != session.plan.execution_plan().plan_hash()
                || id.runtime_implementation_fingerprint()
                    != lane.descriptor().runtime_implementation_fingerprint
                || id.lane_id() != lane.id()
                || program.segments().iter().any(|segment| {
                    segment.end_node_index() as usize
                        > session.plan.execution_plan().payload().nodes().len()
                })
                || catalog.insert(id.clone(), program).is_some()
            {
                return Err(model_error(
                    "causal catalog has a foreign or duplicate program",
                ));
            }
        }
        *self.catalog.write() = Some(Arc::new(VNextReusableExecutionCatalog {
            lane_epoch,
            programs: catalog,
        }));
        Ok(())
    }
}

impl<R: DeviceRuntime> CausalProgramSession<R> {
    /// A replay-enabled program keeps its exact lane across waves. Eager
    /// callers retain the previous per-wave lane lifecycle.
    pub fn execution_lane(&self) -> Result<Arc<ExecutionLane<R>>> {
        if self.installation_only() {
            return Err(model_error(
                "installation-only program has no executable lane",
            ));
        }
        match &self.replay {
            Some(replay) => replay.lane(&self.resources),
            None => self.resources.create_execution_lane().map_err(model_error),
        }
    }

    /// Only the caller's actual continuation waves opt in. Prefill keeps its
    /// ordinary frame and never replays a user's state transition for warmup.
    pub fn continuation_step_request(
        &self,
        request: StepResourceAdmissionRequest,
    ) -> Result<StepResourceAdmissionRequest> {
        if self.replay.is_none() {
            return Ok(request);
        }
        let spans = request
            .work_shape()
            .participant_token_ranges()
            .map_err(model_error)?;
        let rows = spans.first().map(|s| s.immediate_tokens()).unwrap_or(0);
        if !matches!(rows, 1 | 2) || spans.iter().any(|s| s.immediate_tokens() != rows) {
            return Err(model_error(
                "continuation must have uniform one- or two-token rows",
            ));
        }
        let class = ReusableExecutionClassId::new(if rows == 1 {
            UNIFORM_QUERY_REUSABLE_CLASS
        } else {
            TWO_ROW_CLASS
        })
        .map_err(model_error)?;
        let work = request.work_shape().token_view().map_err(model_error)?;
        let bucket = self
            .plan
            .execution_plan()
            .payload()
            .memory()
            .reusable_execution()
            .and_then(|memory| {
                memory.smallest_covering_bucket(
                    &class,
                    work.immediate_sequences(),
                    work.immediate_tokens(),
                    work.immediate_pages(),
                )
            });
        Ok(match bucket {
            Some(bucket) => {
                request.with_reusable_execution_bucket(bucket.bucket().bucket_id().clone())
            }
            None => request,
        })
    }

    pub fn reusable_execution_snapshot(&self) -> serde_json::Value {
        let Some(replay) = &self.replay else {
            return serde_json::json!({"enabled": false, "direct_waves": 0, "fallback_waves": 0});
        };
        let catalog = replay.catalog.read();
        let eager_operations = catalog
            .iter()
            .flat_map(|catalog| catalog.programs.values())
            .flat_map(|program| program.eager_boundary_node_indices())
            .filter_map(|&index| {
                self.plan
                    .execution_plan()
                    .payload()
                    .nodes()
                    .get(index as usize)
            })
            .map(|node| node.operation_id().as_str())
            .collect::<std::collections::BTreeSet<_>>();
        serde_json::json!({
            "enabled": true,
            "direct_waves": replay.direct_waves.load(Ordering::Relaxed),
            "fallback_waves": replay.fallback_waves.load(Ordering::Relaxed),
            "maximum_executables": replay.preparation.maximum_executables(),
            "catalog_programs": catalog.as_ref().map_or(0, |c| c.programs.len()),
            "workspace_buckets": self.plan.execution_plan().payload().memory()
                .reusable_execution().map(|memory| memory.buckets()),
            "eager_boundary_operations": eager_operations,
            "resident_programs": catalog.as_ref().map_or(0, |c| c.programs.values().filter(|p| p.has_resident_segments()).count()),
        })
    }
}
