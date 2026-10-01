//! CPU-only design experiment, deliberately not a production catalog API.
//! Preparation consumes a complete bounded catalog. A lookup returns one
//! numeric record, never a partial `DeviceCostGraphCatalog` or replay authority.
use super::*;
use ferrum_interfaces::execution_cost::{
    LibraryApiNumericWorkV1, LibraryReplayParametersV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedReplayAlgorithmTemplateV1,
};
use ferrum_interfaces::vnext::{
    DeviceBatchingForm, DeviceCostGraphCatalogBuilder, DeviceCostGraphProgram, DeviceDescriptor,
    DeviceNativeOperationId, DeviceRuntime, ExecutionLaneId,
};
use std::collections::BTreeMap;

#[derive(Debug)]
struct Owner {
    device: DeviceDescriptor,
    lane: ExecutionLaneId,
}

#[test]
#[ignore = "CPU timing diagnostic; release CUDA feature build, no GPU/context"]
fn cuda_exact_index_cpu_catalog_build_hit_and_selected_query_cost() {
    use std::{hint::black_box, time::Instant};
    const QUERIES: usize = 512;
    // Generic declared cache capacities. Each slot registers one real
    // three-node program through CudaExecutableCache, with typed Missing
    // gaps and no uploaded GPU executable. No model/hardware lookup occurs.
    for plan in [2, 32, 256].map(|capacity| DeviceReusableExecutionPlan::new(capacity).unwrap()) {
        let setup_started = Instant::now();
        let mut registry = Registry::with_plan(plan);
        let captures = (1..=plan.maximum_executables())
            .map(|slot| registry.capture(slot as u64))
            .collect::<Vec<_>>();
        for capture in &captures {
            registry.register(capture);
        }
        let registry_setup_ns = setup_started.elapsed().as_nanos();
        let nodes = captures
            .iter()
            .map(|capture| capture.node_count() as usize)
            .sum::<usize>();
        let inventory_limits =
            DeviceCostGraphCatalogLimits::new(plan.maximum_executables(), nodes, nodes).unwrap();
        assert!(registry.cache.cost_catalog.get().is_none());

        // Report both preparation stages. The normal shared producer is used
        // once here; the fresh-owned arm below does not imply every old query
        // would have missed its existing shared cache.
        let started = Instant::now();
        let complete = registry
            .cache
            .cost_reusable_graph_catalog_shared(inventory_limits, &mut || Ok(()))
            .unwrap();
        let initial_full_capture_ns = started.elapsed().as_nanos();
        let started = Instant::now();
        let index = ExactIndex::prepare(
            Arc::clone(&registry.owner),
            Arc::clone(&registry.generation),
            Arc::clone(&complete),
            inventory_limits,
            &mut || Ok(()),
        )
        .unwrap();
        let index_preparation_ns = started.elapsed().as_nanos();

        // Semantic checks are outside every timed query loop.
        let original = registry
            .cache
            .cost_reusable_graph_catalog(inventory_limits, &mut || Ok(()))
            .unwrap();
        assert_eq!(&original, complete.as_ref());
        let hit = registry
            .cache
            .cost_reusable_graph_catalog_shared(inventory_limits, &mut || Ok(()))
            .unwrap();
        assert!(Arc::ptr_eq(&hit, &complete));
        assert_eq!(complete.programs().len(), plan.maximum_executables());
        for capture in &captures {
            let record = selected_record(index.query(
                &registry.current(),
                capture.program_id(),
                selected(),
                &mut || Ok(()),
            ));
            let expected = complete
                .programs()
                .iter()
                .find(|program| program.program().program_id() == capture.program_id())
                .unwrap();
            assert_eq!(record.record(), expected);
            assert!(record.is_current(&registry.current()));
            assert!(record.record().uploaded_segments().is_empty());
        }

        let mut fresh_errors = 0usize;
        let started = Instant::now();
        for _ in 0..QUERIES {
            let result = black_box(
                black_box(&registry.cache)
                    .cost_reusable_graph_catalog(black_box(inventory_limits), &mut || Ok(())),
            );
            fresh_errors += usize::from(result.is_err());
            drop(black_box(result));
        }
        let full_fresh_build_total_ns = started.elapsed().as_nanos();

        let mut hit_errors = 0usize;
        let started = Instant::now();
        for _ in 0..QUERIES {
            let result = black_box(
                black_box(&registry.cache).cost_reusable_graph_catalog_shared(
                    black_box(inventory_limits),
                    &mut || Ok(()),
                ),
            );
            hit_errors += usize::from(result.is_err());
            drop(black_box(result));
        }
        let full_cached_hit_total_ns = started.elapsed().as_nanos();

        let mut exact_errors = 0usize;
        let started = Instant::now();
        for iteration in 0..QUERIES {
            let id = captures[iteration % captures.len()].program_id();
            let result = black_box(black_box(&index).query(
                &black_box(&registry).current(),
                black_box(id),
                selected(),
                &mut || Ok(()),
            ));
            exact_errors += usize::from(!matches!(&result, Ok(Lookup::Selected(_))));
            drop(black_box(result));
        }
        let exact_selected_query_total_ns = started.elapsed().as_nanos();
        assert_eq!((fresh_errors, hit_errors, exact_errors), (0, 0, 0));
        assert!(index.is_current(&registry.current()));
        eprintln!(
            "{}",
            serde_json::json!({
                "kind": "cuda_exact_program_index_cpu_comparison",
                "cache_plan_maximum_executables": plan.maximum_executables(),
                "retained_programs": complete.programs().len(),
                "retained_nodes": nodes,
                "retained_uploaded_logical_commands": 0,
                "selected_nodes": captures[0].node_count(),
                "selected_logical_command_limit": selected().maximum_logical_commands(),
                "queries_per_arm": QUERIES,
                "registry_setup_ns": registry_setup_ns,
                "preparation_initial_full_capture_ns": initial_full_capture_ns,
                "preparation_index_from_complete_catalog_ns": index_preparation_ns,
                "preparation_total_ns": initial_full_capture_ns + index_preparation_ns,
                "full_fresh_build_total_ns": full_fresh_build_total_ns,
                "full_fresh_build_mean_ns": full_fresh_build_total_ns as f64 / QUERIES as f64,
                "full_cached_hit_total_ns": full_cached_hit_total_ns,
                "full_cached_hit_mean_ns": full_cached_hit_total_ns as f64 / QUERIES as f64,
                "exact_selected_query_total_ns": exact_selected_query_total_ns,
                "exact_selected_query_mean_ns": exact_selected_query_total_ns as f64 / QUERIES as f64,
                "errors": {"fresh": fresh_errors, "cached": hit_errors, "exact": exact_errors},
                "scope": "CPU numeric registry only; query result destruction included. Fresh build always calls the owned producer for this diagnostic; it is not the old per-query default. Cached hit returns the full Arc without downstream full-catalog validation/program matching; exact lookup includes current owner/generation/stream checks and selected-program limits. No GPU uploads, production index lifecycle, controller budget or SLO claim."
            })
        );
    }
}

/// Pointer identity prevents a recycled generation number from reviving an
/// older snapshot. Only the owning mutation path can mint the next token.
#[derive(Debug)]
struct Generation(u64);

#[derive(Debug, PartialEq, Eq)]
enum Failure {
    Budget,
    RetainedLimit,
    QueryLimit,
    WrongOwner,
    Stale,
}

struct Entry {
    ordinal: usize,
    commands: usize,
}

struct ExactIndex {
    owner: Arc<Owner>,
    generation: Arc<Generation>,
    complete: Arc<DeviceCostGraphCatalog>,
    by_id: BTreeMap<DeviceReusableExecutionProgramId, Entry>,
}

struct SelectedRecord {
    root: Arc<ExactIndex>,
    ordinal: usize,
}
impl SelectedRecord {
    fn record(&self) -> &DeviceCostGraphProgram {
        &self.root.complete.programs()[self.ordinal]
    }
    fn is_current(&self, current: &Current<'_>) -> bool {
        self.root.is_current(current)
    }
}

enum Lookup {
    Unobserved,
    Missing,
    Selected(SelectedRecord),
}

struct Current<'a> {
    owner: &'a Arc<Owner>,
    generation: &'a Arc<Generation>,
    stream_state: DeviceCostGraphStreamState,
}

impl ExactIndex {
    /// The retained inventory bound is a preparation/memory contract. It is
    /// distinct from the work bound charged to one selected program at query.
    /// No partial map escapes any cancellation or identity failure.
    fn prepare(
        owner: Arc<Owner>,
        generation: Arc<Generation>,
        complete: Arc<DeviceCostGraphCatalog>,
        retained: DeviceCostGraphCatalogLimits,
        poll: &mut dyn FnMut() -> Result<(), Failure>,
    ) -> Result<Arc<Self>, Failure> {
        poll()?;
        if !complete.fits(retained) {
            return Err(Failure::RetainedLimit);
        }
        let mut by_id = BTreeMap::new();
        for (ordinal, program) in complete.programs().iter().enumerate() {
            poll()?;
            let id = program.program().program_id();
            if id.lane_id() != owner.lane
                || id.runtime_implementation_fingerprint()
                    != owner.device.runtime_implementation_fingerprint
            {
                return Err(Failure::WrongOwner);
            }
            let mut commands = 0usize;
            for segment in program.uploaded_segments() {
                poll()?;
                commands = commands
                    .checked_add(segment.logical_commands().len())
                    .ok_or(Failure::RetainedLimit)?;
            }
            // The source's private builder already enforces completeness and
            // uniqueness. The index retains that same immutable source root.
            by_id.insert(id.clone(), Entry { ordinal, commands });
        }
        poll()?;
        Ok(Arc::new(Self {
            owner,
            generation,
            complete,
            by_id,
        }))
    }

    fn is_current(&self, current: &Current<'_>) -> bool {
        Arc::ptr_eq(&self.owner, current.owner)
            && Arc::ptr_eq(&self.generation, current.generation)
            && self.complete.stream_state() == current.stream_state
    }

    fn query(
        self: &Arc<Self>,
        current: &Current<'_>,
        id: &DeviceReusableExecutionProgramId,
        selected_work: DeviceCostGraphCatalogLimits,
        poll: &mut dyn FnMut() -> Result<(), Failure>,
    ) -> Result<Lookup, Failure> {
        poll()?;
        if !Arc::ptr_eq(&self.owner, current.owner)
            || id.lane_id() != self.owner.lane
            || id.runtime_implementation_fingerprint()
                != self.owner.device.runtime_implementation_fingerprint
        {
            return Err(Failure::WrongOwner);
        }
        if !self.is_current(current) {
            return Err(Failure::Stale);
        }
        let Some(entry) = self.by_id.get(id) else {
            poll()?;
            return Ok(Lookup::Missing);
        };
        let record = &self.complete.programs()[entry.ordinal];
        if record.program().node_count() as usize > selected_work.maximum_nodes()
            || entry.commands > selected_work.maximum_logical_commands()
        {
            return Err(Failure::QueryLimit);
        }
        poll()?;
        Ok(Lookup::Selected(SelectedRecord {
            root: Arc::clone(self),
            ordinal: entry.ordinal,
        }))
    }
}

struct Registry {
    cache: CudaExecutableCache,
    owner: Arc<Owner>,
    generation: Arc<Generation>,
    prepared: Option<Arc<ExactIndex>>,
}
impl Registry {
    fn new() -> Self {
        Self::with_plan(DeviceReusableExecutionPlan::new(2).unwrap())
    }
    fn with_plan(plan: DeviceReusableExecutionPlan) -> Self {
        let composition = CpuVNextComposition::create(
            DeviceId::new("device.cpu.exact-program-index").unwrap(),
            1024 * 1024,
        )
        .unwrap();
        let lane = ExecutionLane::create(Arc::clone(composition.runtime())).unwrap();
        let owner = Arc::new(Owner {
            device: composition.runtime().descriptor().clone(),
            lane: lane.id(),
        });
        let mut cache = CudaExecutableCache::new();
        cache.configure(plan).unwrap();
        Self {
            cache,
            owner,
            generation: Arc::new(Generation(0)),
            prepared: None,
        }
    }
    fn capture(&self, slot: u64) -> DeviceReusableExecutionCapture {
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("exact-program-index").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        let id = DeviceReusableExecutionProgramId::new(
            serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
            self.owner.device.runtime_implementation_fingerprint.clone(),
            self.owner.lane,
            bucket.bucket_id().clone(),
            "c".repeat(64),
            "d".repeat(64),
            slot,
            1,
            1,
            1,
        )
        .unwrap();
        DeviceReusableExecutionCapture::new(id, 3, vec![], vec![]).unwrap()
    }
    fn mutate(&mut self, change: impl FnOnce(&mut CudaExecutableCache)) {
        // The production follow-up must call this at EVERY real mutation,
        // including same-count upload, sidecar/library change and eviction.
        self.generation = Arc::new(Generation(self.generation.0.checked_add(1).unwrap()));
        self.cache.invalidate_cost_catalog();
        change(&mut self.cache);
    }
    fn register(&mut self, capture: &DeviceReusableExecutionCapture) {
        self.mutate(|cache| register_missing_program(cache, capture));
    }
    fn current(&self) -> Current<'_> {
        Current {
            owner: &self.owner,
            generation: &self.generation,
            stream_state: self.cache.cost_graph_stream_state().unwrap(),
        }
    }
    fn prepare(&mut self) {
        let source = self
            .cache
            .cost_reusable_graph_catalog_shared(retained(), &mut || Ok(()))
            .unwrap();
        self.prepared = Some(
            ExactIndex::prepare(
                Arc::clone(&self.owner),
                Arc::clone(&self.generation),
                source,
                retained(),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    }
    fn query(&self, id: &DeviceReusableExecutionProgramId) -> Result<Lookup, Failure> {
        match &self.prepared {
            None => Ok(Lookup::Unobserved),
            Some(index) => index.query(&self.current(), id, selected(), &mut || Ok(())),
        }
    }
}
fn retained() -> DeviceCostGraphCatalogLimits {
    DeviceCostGraphCatalogLimits::new(4, 12, 12).unwrap()
}
fn selected() -> DeviceCostGraphCatalogLimits {
    DeviceCostGraphCatalogLimits::new(1, 3, 3).unwrap()
}
fn selected_record(result: Result<Lookup, Failure>) -> SelectedRecord {
    match result.unwrap() {
        Lookup::Selected(record) => record,
        _ => panic!("expected an exact numeric program record"),
    }
}

#[test]
fn cuda_two_legal_three_node_programs_fail_aggregate_capture_but_exact_query_is_bounded() {
    let mut registry = Registry::new();
    let a = registry.capture(1);
    let b = registry.capture(2);
    let hot_limits = DeviceCostGraphCatalogLimits::new(2, 5, 5).unwrap();
    registry.register(&a);
    assert_eq!(
        registry
            .cache
            .cost_reusable_graph_catalog_shared(hot_limits, &mut || Ok(()))
            .unwrap()
            .programs()
            .len(),
        1
    );
    registry.register(&b);
    let error = registry
        .cache
        .cost_reusable_graph_catalog_shared(hot_limits, &mut || Ok(()))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("graph catalog node limit exceeded"));
    assert!(registry.cache.cost_catalog.get().is_none());
    // Preparation retains both programs. The query is charged only 3 nodes;
    // no full catalog with a forged resident-program count is constructed.
    registry.prepare();
    for id in [a.program_id(), b.program_id()] {
        let record = selected_record(registry.query(id));
        assert_eq!(record.record().program().node_count(), 3);
        assert_eq!(record.record().program().program_id(), id);
        assert_eq!(record.root.complete.programs().len(), 2);
        assert!(record.is_current(&registry.current()));
    }
}

#[test]
fn cuda_exact_index_distinguishes_unobserved_absence_and_evicted_numeric_records() {
    let mut registry = Registry::new();
    let a = registry.capture(1);
    assert!(matches!(
        registry.query(a.program_id()),
        Ok(Lookup::Unobserved)
    ));
    registry.prepare();
    assert!(matches!(
        registry.query(a.program_id()),
        Ok(Lookup::Missing)
    ));
    registry.register(&a);
    registry.prepare();
    let before = selected_record(registry.query(a.program_id()));
    registry.mutate(|cache| {
        cache.programs.get_mut(a.program_id()).unwrap().descriptor =
            DeviceReusableExecutionProgram::new(
                &a,
                vec![],
                vec![],
                (0..3)
                    .map(|n| {
                        DeviceReusableExecutionProgramGap::new(
                            n,
                            DeviceReusableExecutionProgramGapReason::Evicted,
                        )
                    })
                    .collect(),
            )
            .unwrap();
    });
    assert!(!before.is_current(&registry.current()));
    assert!(matches!(
        registry.query(a.program_id()),
        Err(Failure::Stale)
    ));
    registry.prepare();
    let after = selected_record(registry.query(a.program_id()));
    assert!(after
        .record()
        .program()
        .gaps()
        .iter()
        .all(|g| g.reason() == DeviceReusableExecutionProgramGapReason::Evicted));
    assert!(after.record().uploaded_segments().is_empty());
    assert!(before
        .record()
        .program()
        .gaps()
        .iter()
        .all(|g| g.reason() != DeviceReusableExecutionProgramGapReason::Evicted));
}

#[test]
fn cuda_exact_index_requires_owner_full_identity_generation_and_current_graph_state() {
    let mut registry = Registry::new();
    let capture = registry.capture(1);
    registry.register(&capture);
    registry.prepare();
    let root = Arc::clone(registry.prepared.as_ref().unwrap());
    let other = Registry::new();
    assert!(matches!(
        root.query(
            &other.current(),
            capture.program_id(),
            selected(),
            &mut || Ok(())
        ),
        Err(Failure::WrongOwner)
    ));
    let id = capture.program_id();
    let different_layout = DeviceReusableExecutionProgramId::new(
        id.plan_hash().clone(),
        id.runtime_implementation_fingerprint().to_owned(),
        id.lane_id(),
        id.bucket_id().clone(),
        "e".repeat(64),
        id.lane_stable_layout_fingerprint().to_owned(),
        id.lane_slot_id(),
        id.immediate_sequences(),
        id.immediate_tokens(),
        id.immediate_pages(),
    )
    .unwrap();
    assert!(matches!(
        registry.query(&different_layout),
        Ok(Lookup::Missing)
    ));
    let different_topology = id.clone().with_topology_fingerprint(
        ferrum_interfaces::vnext::DeviceReusableExecutionTopologyFingerprint::from_sha256([3; 32]),
    );
    assert!(matches!(
        registry.query(&different_topology),
        Ok(Lookup::Missing)
    ));
    let same_number = Arc::new(Generation(registry.generation.0));
    let current = Current {
        generation: &same_number,
        ..registry.current()
    };
    assert!(matches!(
        root.query(&current, id, selected(), &mut || Ok(())),
        Err(Failure::Stale)
    ));
    let current = Current {
        stream_state: DeviceCostGraphStreamState::new(
            DeviceCostGraphConfiguration::StartupReady,
            0,
            1,
            0,
        )
        .unwrap(),
        ..registry.current()
    };
    assert!(matches!(
        root.query(&current, id, selected(), &mut || Ok(())),
        Err(Failure::Stale)
    ));
}

#[test]
fn cuda_exact_index_limits_and_cancellation_never_publish_partial_absence() {
    let mut registry = Registry::new();
    let a = registry.capture(1);
    let b = registry.capture(2);
    registry.register(&a);
    registry.register(&b);
    let source = registry
        .cache
        .cost_reusable_graph_catalog_shared(retained(), &mut || Ok(()))
        .unwrap();
    assert!(matches!(
        ExactIndex::prepare(
            Arc::clone(&registry.owner),
            Arc::clone(&registry.generation),
            Arc::clone(&source),
            selected(),
            &mut || Ok(())
        ),
        Err(Failure::RetainedLimit)
    ));
    let mut remaining = 2;
    assert!(matches!(
        ExactIndex::prepare(
            Arc::clone(&registry.owner),
            Arc::clone(&registry.generation),
            source,
            retained(),
            &mut || {
                if remaining == 0 {
                    Err(Failure::Budget)
                } else {
                    remaining -= 1;
                    Ok(())
                }
            }
        ),
        Err(Failure::Budget)
    ));
    assert!(matches!(
        registry.query(a.program_id()),
        Ok(Lookup::Unobserved)
    ));
    registry.prepare();
    let root = registry.prepared.as_ref().unwrap();
    assert!(matches!(
        root.query(
            &registry.current(),
            a.program_id(),
            DeviceCostGraphCatalogLimits::new(1, 2, 3).unwrap(),
            &mut || Ok(())
        ),
        Err(Failure::QueryLimit)
    ));
    assert!(matches!(
        root.query(
            &registry.current(),
            a.program_id(),
            selected(),
            &mut || Err(Failure::Budget)
        ),
        Err(Failure::Budget)
    ));
    assert!(matches!(
        registry.query(b.program_id()),
        Ok(Lookup::Selected(_))
    ));
    let absent = registry.capture(3);
    let mut budget_available = true;
    assert!(matches!(
        root.query(
            &registry.current(),
            absent.program_id(),
            selected(),
            &mut || {
                if std::mem::replace(&mut budget_available, false) {
                    Ok(())
                } else {
                    Err(Failure::Budget)
                }
            }
        ),
        Err(Failure::Budget)
    ));
}

fn uploaded_inventory(
    registry: &Registry,
    capture: &DeviceReusableExecutionCapture,
    fp: char,
    tokens: u64,
    library_policy: u8,
) -> Arc<DeviceCostGraphCatalog> {
    let descriptor = DeviceReusableExecutionProgram::new(
        capture,
        vec![DeviceReusableExecutionSegment::new(0, 0, 3, 3).unwrap()],
        vec![],
        vec![],
    )
    .unwrap();
    let mut builder = DeviceCostGraphCatalogBuilder::new(
        DeviceCostGraphStreamState::new(DeviceCostGraphConfiguration::StartupReady, 1, 1, 0)
            .unwrap(),
        retained(),
    )
    .unwrap();
    builder.push_program(&descriptor, &mut || Ok(())).unwrap();
    let mut evidence = SelectedCommandCostBuilderV1::new_with_algorithm_work(tokens);
    evidence
        .library_call_with_replay_parameters(
            SelectedAlgorithmClassV1::library_api(
                "test.vendor.GemmEx",
                1,
                [library_policy; 32],
                [2; 32],
            )
            .unwrap(),
            LibraryApiNumericWorkV1 {
                output_elements: tokens * 64,
                reduction_units_per_output: 32,
            },
            LibraryReplayParametersV1 {
                fixed_parameters: &[tokens, 64, 32],
            },
        )
        .unwrap();
    let template =
        SelectedReplayAlgorithmTemplateV1::from_selected(&evidence.finish().unwrap(), tokens, 1, 0)
            .unwrap();
    let rows: Vec<_> = (0..3)
        .map(|n| {
            DeviceReplayedLogicalCommandAttribution::new(
                n,
                n,
                DeviceNativeOperationId::new("test.exact-index.kernel").unwrap(),
                DeviceBatchingForm::Scalar,
                1,
                tokens,
                1,
                0,
                1,
            )
            .unwrap()
            .with_captured_replay_template(Some(template))
        })
        .collect();
    builder
        .push_uploaded_segment(
            &descriptor.segments()[0],
            &fp.to_string().repeat(64),
            &rows,
            &mut || Ok(()),
        )
        .unwrap();
    let result = Arc::new(builder.finish(&mut || Ok(())).unwrap());
    assert_eq!(
        result.programs()[0].program().program_id().lane_id(),
        registry.owner.lane
    );
    result
}

#[test]
fn cuda_exact_index_same_count_metadata_replacement_invalidates_old_root_without_mutating_it() {
    let registry = Registry::new();
    let capture = registry.capture(1);
    let old_source = uploaded_inventory(&registry, &capture, 'e', 1, 1);
    let new_source = uploaded_inventory(&registry, &capture, 'f', 2, 1);
    assert_eq!(old_source.stream_state(), new_source.stream_state());
    let old_generation = Arc::new(Generation(1));
    let new_generation = Arc::new(Generation(2));
    let old = ExactIndex::prepare(
        Arc::clone(&registry.owner),
        Arc::clone(&old_generation),
        old_source,
        retained(),
        &mut || Ok(()),
    )
    .unwrap();
    let new = ExactIndex::prepare(
        Arc::clone(&registry.owner),
        Arc::clone(&new_generation),
        new_source,
        retained(),
        &mut || Ok(()),
    )
    .unwrap();
    let current = Current {
        owner: &registry.owner,
        generation: &old_generation,
        stream_state: old.complete.stream_state(),
    };
    let selected_old =
        selected_record(old.query(&current, capture.program_id(), selected(), &mut || Ok(())));
    let current = Current {
        generation: &new_generation,
        ..current
    };
    assert!(!selected_old.is_current(&current));
    assert!(matches!(
        old.query(&current, capture.program_id(), selected(), &mut || Ok(())),
        Err(Failure::Stale)
    ));
    let selected_new =
        selected_record(new.query(&current, capture.program_id(), selected(), &mut || Ok(())));
    let a = &selected_old.record().uploaded_segments()[0];
    let b = &selected_new.record().uploaded_segments()[0];
    assert_ne!(
        a.reusable_executable_fingerprint(),
        b.reusable_executable_fingerprint()
    );
    assert_eq!(a.logical_commands()[0].token_count(), 1);
    assert_eq!(b.logical_commands()[0].token_count(), 2);
}

#[test]
fn cuda_exact_index_library_template_change_cannot_hide_behind_equal_execution_counts() {
    let registry = Registry::new();
    let capture = registry.capture(1);
    let old_source = uploaded_inventory(&registry, &capture, 'e', 1, 1);
    let changed_source = uploaded_inventory(&registry, &capture, 'e', 1, 2);
    // Execution Eq deliberately excludes passive cost templates. It cannot
    // decide whether a prepared numeric root is safe to retain.
    assert_eq!(old_source, changed_source);
    let row = |source: &DeviceCostGraphCatalog| {
        source.programs()[0].uploaded_segments()[0].logical_commands()[0].clone()
    };
    assert!(!row(&old_source).same_cost_catalog_metadata(&row(&changed_source)));
    let old_generation = Arc::new(Generation(1));
    let new_generation = Arc::new(Generation(2));
    let old = ExactIndex::prepare(
        Arc::clone(&registry.owner),
        Arc::clone(&old_generation),
        old_source,
        retained(),
        &mut || Ok(()),
    )
    .unwrap();
    let new = ExactIndex::prepare(
        Arc::clone(&registry.owner),
        Arc::clone(&new_generation),
        changed_source,
        retained(),
        &mut || Ok(()),
    )
    .unwrap();
    let old_current = Current {
        owner: &registry.owner,
        generation: &old_generation,
        stream_state: old.complete.stream_state(),
    };
    let selected_old = selected_record(old.query(
        &old_current,
        capture.program_id(),
        selected(),
        &mut || Ok(()),
    ));
    let current = Current {
        generation: &new_generation,
        ..old_current
    };
    assert!(!selected_old.is_current(&current));
    assert!(matches!(
        old.query(&current, capture.program_id(), selected(), &mut || Ok(())),
        Err(Failure::Stale)
    ));
    let selected_new =
        selected_record(new.query(&current, capture.program_id(), selected(), &mut || Ok(())));
    assert!(
        !selected_old.record().uploaded_segments()[0].logical_commands()[0]
            .same_cost_catalog_metadata(
                &selected_new.record().uploaded_segments()[0].logical_commands()[0]
            )
    );
    assert!(matches!(
        new.query(
            &current,
            capture.program_id(),
            DeviceCostGraphCatalogLimits::new(1, 3, 2).unwrap(),
            &mut || Ok(())
        ),
        Err(Failure::QueryLimit)
    ));
}
