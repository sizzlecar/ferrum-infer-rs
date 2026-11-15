use super::*;

fn time_limits() -> FastH3TimeConditioningLimits {
    FastH3TimeConditioningLimits {
        maximum_time_rows: 64,
        maximum_metadata_bytes: 1 << 20,
        maximum_stored_bytes: 1 << 20,
    }
}
fn output_limits() -> FastH3OutputConditioningLimits {
    FastH3OutputConditioningLimits {
        maximum_rows: 4096,
        maximum_time_rows: 64,
        maximum_metadata_bytes: 1 << 20,
        maximum_stored_bytes: 1 << 20,
    }
}
pub(super) fn final_limits() -> FastH3FinalOutputLimits {
    FastH3FinalOutputLimits {
        conditioning: output_limits(),
        projection: FastH3PddProjectionLimits {
            maximum_rows: 4096,
            maximum_metadata_bytes: 1 << 20,
            maximum_source_bytes: 1 << 20,
            maximum_fused_bytes: 1 << 20,
        },
        maximum_metadata_bytes: 1 << 20,
        maximum_plan_nodes: 128,
        maximum_stored_bytes: 1 << 20,
        maximum_pdd_steps: 64,
    }
}
pub(super) fn plan(p: &FastH3TransformerPackage) -> Arc<FastH3PddPlan> {
    let c = FastH3Checkpoint::parse(
        include_bytes!("../../../../fixtures/inference.json"),
        include_bytes!("../../../../fixtures/transformer.json"),
        include_bytes!("../../../../fixtures/video_scheduler.json"),
        include_bytes!("../../../../fixtures/audio_scheduler.json"),
        FastH3MetadataLimits {
            max_file_bytes: 4096,
            max_pdd_steps: 64,
        },
    )
    .unwrap();
    let original = FastH3PddPlan::new(&c).unwrap();
    Arc::new(
        FastH3PddPlan::from_schedule(
            original.schedule().clone(),
            p.config().video_head_width(),
            p.config().audio_channels(),
            64,
        )
        .unwrap(),
    )
}
pub(super) fn tables(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    p: &FastH3TransformerPackage,
    schedule: &FastH3ConditioningSchedule,
) -> (
    Arc<[FastH3ConditioningTable]>,
    Arc<FastH3OutputConditioningTable>,
) {
    let mut tables = Vec::new();
    for block in 0..p.config().layers() {
        let c = FastH3TimeConditioningConfig::from_package(
            p,
            block,
            schedule.timesteps().len() as u64,
            time_limits(),
        )
        .unwrap();
        let source = FastH3TimeConditioningSource::new(p, &c, h, time_limits()).unwrap();
        let out = run(
            r,
            h,
            FastH3TimeConditioningFamily::new(time_limits()).unwrap(),
            &c,
            FastH3TimeConditioningFamily::PROFILE_ID,
            c.compile_options().unwrap(),
            ComponentWeightSource::Stored(&source),
            vec![c.managed_schedule_features(h, schedule).unwrap()],
        );
        tables.push(
            FastH3ConditioningTable::from_execution(c, schedule, out, time_limits()).unwrap(),
        );
    }
    let c = FastH3OutputTimeConfig::from_package(p, schedule, output_limits()).unwrap();
    let source =
        FastH3OutputConditioningSource::for_time_projection(p, &c, output_limits()).unwrap();
    let out = run(
        r,
        h,
        FastH3OutputTimeFamily::new(output_limits()).unwrap(),
        &c,
        FastH3OutputTimeFamily::PROFILE_ID,
        c.compile_options().unwrap(),
        ComponentWeightSource::Stored(&source),
        vec![c.embedding_input(&tables[0], h).unwrap()],
    );
    (
        tables.into(),
        Arc::new(FastH3OutputConditioningTable::from_execution(c, out, output_limits()).unwrap()),
    )
}
pub(super) fn request(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    c: &FastH3DenoisingConfig,
    layout: &FastH3PackedLayout,
    plan: Arc<FastH3PddPlan>,
    tables: Arc<[FastH3ConditioningTable]>,
    out: Arc<FastH3OutputConditioningTable>,
    salt: usize,
) -> FastH3DenoisingRequest {
    request_inner(r, h, c, layout, plan, tables, out, salt, None)
}
fn request_inner(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    c: &FastH3DenoisingConfig,
    layout: &FastH3PackedLayout,
    plan: Arc<FastH3PddPlan>,
    tables: Arc<[FastH3ConditioningTable]>,
    out: Arc<FastH3OutputConditioningTable>,
    salt: usize,
    prepared: Option<&FastH3DenoisingBatchPlan>,
) -> FastH3DenoisingRequest {
    let specs = c.compile_options(limits_batch()).unwrap();
    let tensor = |name: &str| {
        let s = &specs.tensor_specs()[&id(name)];
        let mut x = input(
            h,
            name,
            s.dimensions.iter().take(s.dimensions.len() - 1).product(),
            *s.dimensions.last().unwrap(),
            s.element_type,
            salt,
        );
        if s.dimensions.len() == 3 {
            x = ManagedHostTensor::new(
                x.value_id().clone(),
                ResolvedTensorSpec::new(s.dimensions.clone(), s.element_type, s.layout.clone())
                    .unwrap(),
                x.bytes().clone(),
            )
            .unwrap();
        }
        x
    };
    let text = tensor(FastH3DenoisingFamily::TEXT_INPUT_ID);
    // Explicit diagnostic refined rows, produced by a real managed copy. The
    // test verifies batch adaptation, not text-refiner or full-DiT numerics.
    let refined = completed::output(
        r,
        h,
        FastH3TextRefinerFamily::OUTPUT_ID,
        3,
        c.backbone().hidden_size(),
        text.bytes().bytes(),
    );
    if let Some(prepared) = prepared {
        return FastH3DenoisingRequest::initialize_from_plan(
            prepared,
            RequestIdentity::new(format!("request.batch.{salt}")).unwrap(),
            plan,
            layout,
            h.clone(),
            &tensor(FastH3DenoisingFamily::VIDEO_INPUT_ID),
            &tensor(FastH3DenoisingFamily::AUDIO_INPUT_ID),
            &refined,
            [
                tensor(FastH3DenoisingFamily::COSINE_ID),
                tensor(FastH3DenoisingFamily::SINE_ID),
            ],
            tables,
            out,
        )
        .unwrap();
    }
    FastH3DenoisingRequest::initialize(
        c.clone(),
        limits_batch(),
        RequestIdentity::new(format!("request.batch.{salt}")).unwrap(),
        plan,
        layout,
        h.clone(),
        &tensor(FastH3DenoisingFamily::VIDEO_INPUT_ID),
        &tensor(FastH3DenoisingFamily::AUDIO_INPUT_ID),
        &refined,
        [
            tensor(FastH3DenoisingFamily::COSINE_ID),
            tensor(FastH3DenoisingFamily::SINE_ID),
        ],
        tables,
        out,
    )
    .unwrap()
}
pub(super) fn prediction(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    p: &FastH3TransformerPackage,
    layout: &FastH3PackedLayout,
    plan: &FastH3PddPlan,
    request: &FastH3DenoisingRequest,
    hidden: &TensorExecutionOutput,
) -> Vec<TensorExecutionOutput> {
    let (c, source) =
        FastH3FinalOutputSource::prepare(p, plan, layout, request.evaluation(), h, final_limits())
            .unwrap();
    run(
        r,
        h,
        FastH3FinalOutputFamily::new(final_limits()).unwrap(),
        &c,
        FastH3FinalOutputFamily::PROFILE_ID,
        c.compile_options(final_limits()).unwrap(),
        ComponentWeightSource::Stored(&source),
        request.final_output_inputs(&c, hidden).unwrap(),
    )
}
fn normalized(
    r: &ReferenceVNextComposition,
    h: &HostMemoryAccount,
    p: &FastH3TransformerPackage,
    c: &FastH3ModulationConfig,
    packed: &TensorExecutionOutput,
    table: &ManagedHostTensor,
) -> TensorExecutionOutput {
    let limits = limits_batch().backbone.block.modulation;
    let source = FastH3ModulationSource::new(p, c, limits).unwrap();
    run(
        r,
        h,
        FastH3ModulationFamily::new(limits).unwrap(),
        c,
        FastH3ModulationFamily::PROFILE_ID,
        c.compile_options().unwrap(),
        ComponentWeightSource::Stored(&source),
        vec![
            copy(h, packed, FastH3ModulationFamily::INPUT_ID),
            ManagedHostTensor::new(
                id(FastH3ModulationFamily::TABLE_ID),
                table.tensor().clone(),
                table.bytes().clone(),
            )
            .unwrap(),
        ],
    )
    .into_iter()
    .find(|o| o.value_id().as_str() == FastH3ModulationFamily::OUTPUT_ID)
    .unwrap()
}

#[test]
fn mixed_evaluation_batch_preserves_independent_members_and_atomic_publication() {
    let (_dir, p) = package();
    let layout = super::super::super::attention_core::layout();
    let h = host();
    let l = limits_batch();
    let tags = tags();
    let plan = plan(&p);
    let schedule = FastH3ConditioningSchedule::all_modalities(&plan).unwrap();
    let (one, s1, regions1) = FastH3DenoisingSource::prepare(
        &p,
        &layout,
        &schedule,
        &tags,
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        l,
    )
    .unwrap();
    let (two, s2, regions2) = FastH3DenoisingSource::prepare_batch(
        &p,
        &layout,
        &schedule,
        &[&tags, &tags],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        l,
    )
    .unwrap();
    let batch_plan = FastH3DenoisingBatchPlan::prepare(two.clone(), l, regions2.clone()).unwrap();
    let runtime = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.reference.batch_continuations").unwrap(),
        128 << 20,
    )
    .unwrap();
    let (tables, output_table) = tables(&runtime, &h, &p, &schedule);
    let mut a = request(
        &runtime,
        &h,
        &one,
        layout.packed(),
        plan.clone(),
        tables.clone(),
        output_table.clone(),
        0,
    )
    .with_batch_regions(regions1.clone())
    .unwrap();
    let single_plan = FastH3DenoisingBatchPlan::prepare(one.clone(), l, regions1.clone()).unwrap();
    // This independent member uses the service's cold-plan initializer while
    // the control member above retains the original B1 API.
    let mut b = request_inner(
        &runtime,
        &h,
        &one,
        layout.packed(),
        plan.clone(),
        tables.clone(),
        output_table.clone(),
        1,
        Some(&single_plan),
    );
    let rows = one.backbone().packed_rows();
    let cols = one.backbone().hidden_size();
    let hidden = [0, 1].map(|salt| {
        let x = input(
            &h,
            "value.diag.hidden",
            rows,
            cols,
            ElementType::Bf16,
            salt + 4,
        );
        completed::output(
            &runtime,
            &h,
            FastH3DenoisingFamily::OUTPUT_ID,
            rows,
            cols,
            x.bytes().bytes(),
        )
    });
    b.advance(&prediction(
        &runtime,
        &h,
        &p,
        layout.packed(),
        &plan,
        &b,
        &hidden[1],
    ))
    .unwrap();
    assert_eq!((a.evaluation(), b.evaluation()), (0, 1));
    let singles = [a.denoising_inputs().unwrap(), b.denoising_inputs().unwrap()];
    let token = batch_plan.bind(vec![&mut a, &mut b]).unwrap();
    let original = token.inputs().unwrap();
    let joined_bytes = original
        .iter()
        .map(|v| h.backing_bytes(v.bytes().len()).unwrap())
        .sum::<usize>();
    drop(original);
    let before = h.usage().unwrap().allocated_bytes();
    let short = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, joined_bytes - 1)])
        .unwrap();
    assert!(token.inputs_in(&short.account()).is_err());
    assert_eq!(h.usage().unwrap().allocated_bytes(), before);
    short.close().unwrap();
    assert!(token.inputs_in(&host()).is_err());
    let batch_scope = h
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, joined_bytes)])
        .unwrap();
    let joined = token.inputs_in(&batch_scope.account()).unwrap();
    // Release unused credit; actual joined buffers retain their original charge.
    batch_scope.close().unwrap();
    let table_id = one.backbone().table_input_id(0).unwrap();
    let get = |v: &[ManagedHostTensor]| {
        v.iter()
            .find(|x| x.value_id() == &table_id)
            .unwrap()
            .bytes()
            .bytes()
            .to_vec()
    };
    assert_ne!(get(&singles[0]), get(&singles[1]));
    assert_eq!(get(&joined), [get(&singles[0]), get(&singles[1])].concat());
    let packed = project_pack(&runtime, &h, &p, layout.packed(), &singles);
    let norms = singles
        .iter()
        .map(|member| {
            let x = project_pack(
                &runtime,
                &h,
                &p,
                layout.packed(),
                std::slice::from_ref(member),
            );
            normalized(
                &runtime,
                &h,
                &p,
                one.backbone().blocks()[0].modulation_config(),
                &x,
                member.iter().find(|v| v.value_id() == &table_id).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let norm = normalized(
        &runtime,
        &h,
        &p,
        two.backbone().blocks()[0].modulation_config(),
        &packed,
        joined.iter().find(|v| v.value_id() == &table_id).unwrap(),
    );
    assert_eq!(
        norm.bytes().bytes(),
        norms
            .iter()
            .flat_map(|o| o.bytes().bytes().iter().copied())
            .collect::<Vec<_>>()
    );
    drop(token);
    // The same independent continuations now change slots. No reconstruction
    // or solver advance is needed to prepare this next exact membership.
    let token = batch_plan.bind(vec![&mut b, &mut a]).unwrap();
    assert_eq!(token.member(0).unwrap().1, 1);
    assert_eq!(token.member(1).unwrap().1, 0);
    let bytes = [hidden[1].bytes().bytes(), hidden[0].bytes().bytes()].concat();
    let full = completed::output(
        &runtime,
        &h,
        FastH3DenoisingFamily::OUTPUT_ID,
        2 * rows,
        cols,
        &bytes,
    );
    let mut predictions = Vec::new();
    for (slot, evaluation) in [1, 0].into_iter().enumerate() {
        let (c, source) = FastH3FinalOutputSource::prepare(
            &p,
            &plan,
            layout.packed(),
            evaluation,
            &h,
            final_limits(),
        )
        .unwrap();
        let input = token.final_output_inputs(slot, &c, &full).unwrap();
        assert_eq!(input[0].bytes().bytes(), hidden[1 - slot].bytes().bytes());
        assert_eq!(input[0].bytes().class(), HostMemoryClass::Output);
        assert_eq!(
            input[0].bytes().bytes().as_ptr(),
            full.bytes().bytes()[slot * hidden[1 - slot].bytes().len()..].as_ptr()
        );
        assert!(token.final_output_inputs(1 - slot, &c, &full).is_err());
        predictions.push(run(
            &runtime,
            &h,
            FastH3FinalOutputFamily::new(final_limits()).unwrap(),
            &c,
            FastH3FinalOutputFamily::PROFILE_ID,
            c.compile_options(final_limits()).unwrap(),
            ComponentWeightSource::Stored(&source),
            input,
        ));
    }
    drop(token);
    let states = [
        a.video().bytes().bytes().to_vec(),
        a.audio().bytes().bytes().to_vec(),
        b.video().bytes().bytes().to_vec(),
        b.audio().bytes().bytes().to_vec(),
    ];
    let retained = h.usage().unwrap().allocated_bytes();
    let bad = vec![predictions[0].clone(), vec![predictions[1][0].clone()]];
    assert!(batch_plan
        .bind(vec![&mut b, &mut a])
        .unwrap()
        .advance(&bad)
        .is_err());
    assert_eq!((a.evaluation(), b.evaluation()), (0, 1));
    assert_eq!(
        states,
        [
            a.video().bytes().bytes().to_vec(),
            a.audio().bytes().bytes().to_vec(),
            b.video().bytes().bytes().to_vec(),
            b.audio().bytes().bytes().to_vec()
        ]
    );
    assert_eq!(h.usage().unwrap().allocated_bytes(), retained);
    batch_plan
        .bind(vec![&mut b, &mut a])
        .unwrap()
        .advance(&predictions)
        .unwrap();
    assert_eq!((a.evaluation(), b.evaluation()), (1, 2));
    // A same-shaped but different physical source interpretation is refused;
    // this does not authenticate payload content or replace Core installation.
    let mut wire = serde_json::to_value(&two).unwrap();
    wire["media"]["components"][0]["external_names"] = serde_json::json!(["another.actual.source"]);
    let other = FastH3DenoisingBatchPlan::prepare(
        serde_json::from_value(wire).unwrap(),
        l,
        regions2.clone(),
    )
    .unwrap();
    assert!(other.bind(vec![&mut a, &mut b]).is_err());
    let changed_tags = [FastH3TextTagRun {
        rows: 3,
        tag: FastH3TextTokenTag::Text,
    }];
    let (wrong_tags, wrong_source, wrong_regions) = FastH3DenoisingSource::prepare_batch(
        &p,
        &layout,
        &schedule,
        &[&changed_tags, &tags],
        FastH3VsaSelection::Trained,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        l,
    )
    .unwrap();
    let wrong_tags =
        FastH3DenoisingBatchPlan::prepare(wrong_tags, l, wrong_regions.clone()).unwrap();
    assert!(wrong_tags.bind(vec![&mut a, &mut b]).is_err());
    drop((wrong_source, wrong_regions));
    let (dense, dense_source, dense_regions) = FastH3DenoisingSource::prepare_batch(
        &p,
        &layout,
        &schedule,
        &[&tags, &tags],
        FastH3VsaSelection::Dense,
        FastH3PrefixAttentionArithmetic::ReferenceSequential,
        &h,
        l,
    )
    .unwrap();
    let dense = FastH3DenoisingBatchPlan::prepare(dense, l, dense_regions.clone()).unwrap();
    assert!(dense.bind(vec![&mut a, &mut b]).is_err());
    for target in ["normalization", "qk"] {
        let mut wire = serde_json::to_value(&two).unwrap();
        let block = &mut wire["backbone"]["blocks"][0];
        let epsilon = if target == "normalization" {
            &mut block["norm1"]["epsilon"]
        } else {
            &mut block["attention_input"]["qk"]["epsilon"]
        };
        *epsilon = serde_json::to_value(CanonicalRational::new(1, 1000).unwrap()).unwrap();
        let different = FastH3DenoisingBatchPlan::prepare(
            serde_json::from_value(wire).unwrap(),
            l,
            regions2.clone(),
        )
        .unwrap();
        assert!(different.bind(vec![&mut a, &mut b]).is_err());
    }
    drop((
        batch_plan,
        single_plan,
        other,
        wrong_tags,
        dense,
        dense_source,
        dense_regions,
    ));
    drop((
        a,
        b,
        tables,
        output_table,
        singles,
        joined,
        packed,
        norm,
        norms,
        hidden,
        full,
        predictions,
        bad,
        s1,
        s2,
        regions1,
        regions2,
    ));
    assert_eq!(h.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(runtime.runtime().snapshot().live_allocations, 0);
}
