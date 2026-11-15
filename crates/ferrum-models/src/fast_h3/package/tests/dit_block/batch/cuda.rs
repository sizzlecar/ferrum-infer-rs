//! Full media projection + packing + two actual DiT blocks on one CUDA runtime.
//! Reference Time/PDD heads surround the GPU forward so this isolates batching;
//! it is not a second end-to-end sampling runner or a service/quality benchmark.
use super::super::super::attention_core::cuda_factory;
use super::*;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationReceipt, TensorComponentSession,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use std::sync::atomic::Ordering;
mod support;
use support::*;

#[test]
#[ignore = "requires SM120 Sage/NVFP4 lock and explicit cuDNN; same-runtime B1/B2 shared installation, <=2 BF16 ULP batch equivalence; not product CB"]
fn denoising_cuda_shared_b1_b2_mixed_evaluations_rebind_and_commit() {
    let host = HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            256 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 96 << 20,
                output_bytes: 64 << 20,
                transfer_staging_bytes: 32 << 20,
                weight_backing_bytes: 64 << 20,
            },
        )
        .unwrap(),
        CudaDeviceRuntime::pinned_weight_upload_alignment().unwrap(),
    )
    .unwrap();
    let mut maximum_ulp = 0;
    let uploaded_components;
    let installed_bytes;
    {
        let (_dir, p) = support::fixture();
        let layout = super::super::super::attention_core::layout();
        let limits = limits_batch();
        let tags = tags();
        let plan = continuations::plan(&p);
        let schedule = FastH3ConditioningSchedule::all_modalities(&plan).unwrap();
        let prepare = |tags: &[&[FastH3TextTagRun]]| {
            FastH3DenoisingSource::prepare_batch(
                &p,
                &layout,
                &schedule,
                tags,
                FastH3VsaSelection::Trained,
                FastH3PrefixAttentionArithmetic::VendorFused,
                &host,
                limits,
            )
            .unwrap()
        };
        let (one, source, regions1) = prepare(&[&tags]);
        let (two, _second_source, regions2) = prepare(&[&tags, &tags]);
        let batch_plan = FastH3DenoisingBatchPlan::prepare(two, limits, regions2.clone()).unwrap();
        let reference = ReferenceVNextComposition::create_with_capacity(
            DeviceId::new("device.reference.batch_cuda_conditioning").unwrap(),
            128 << 20,
        )
        .unwrap();
        let (tables, output_table) = continuations::tables(&reference, &host, &p, &schedule);
        let make_request = |salt| {
            continuations::request(
                &reference,
                &host,
                &one,
                layout.packed(),
                plan.clone(),
                tables.clone(),
                output_table.clone(),
                salt,
            )
            .with_batch_regions(regions1.clone())
            .unwrap()
        };
        let mut a = make_request(0);
        let mut b = make_request(1);
        let mut control_a = make_request(0);
        let mut control_b = make_request(1);
        let mut other_partner = make_request(2);

        let prepare_family = |config: &FastH3DenoisingConfig| {
            io::prepare(
                FastH3DenoisingFamily::new(limits).unwrap(),
                config,
                config.profile_id(),
            )
        };
        let prepared1 = prepare_family(&one);
        let prepared2 = prepare_family(batch_plan.configuration());
        // TensorPlanBinding::input_uploads supplies one upload per complete
        // managed input; dispatch prepends these to the provider compute nodes
        // in the SAME wave. This fixed contiguous/stateless fixture has no
        // zero-initialized state, and its providers use OverwriteBeforeRead
        // scratch and one compute command each. Readback is synchronous after
        // the fence, outside this wave. Retain the separate 128-command static
        // upload batch allowance as well.
        let maximum_commands = [&prepared1, &prepared2]
            .into_iter()
            .map(|p| {
                let compute = p.program()
                    .blocks()
                    .iter()
                    .map(|b| b.nodes.len())
                    .sum::<usize>();
                let inputs = p.program().inputs().len();
                let total = compute.checked_add(inputs).unwrap();
                eprintln!("B1/B2 CUDA wave budget: {compute} compute + {inputs} managed input uploads = {total} commands");
                total
            })
            .max()
            .unwrap()
            .max(128);
        let runtime = cuda_factory::runtime_with_command_budget(maximum_commands);
        let providers = cuda_factory::SharedProviders::new(runtime.clone(), &operations());
        let registry1 = providers.registry(regions1.clone(), contracts());
        let registry2 = providers.registry(regions2.clone(), contracts());
        let compiled1 = compile(
            &runtime,
            &registry1,
            prepared1,
            &one.compile_options(limits).unwrap(),
        );
        let compiled2 = compile(
            &runtime,
            &registry2,
            prepared2,
            batch_plan.compile_options(),
        );
        let counted = io::Counting {
            source: &source,
            reads: 0.into(),
        };
        let (mut single, weights) = compiled1
            .initialize_shared_weights(
                runtime.clone(),
                host.clone(),
                ComponentWeightSource::Stored(&counted),
                initial("b1"),
            )
            .unwrap_or_else(|error| {
                let message = format!("{error:?}");
                std::mem::forget(error);
                panic!("B1 shared installation rejected: {message}")
            });
        uploaded_components = counted.reads.load(Ordering::SeqCst);
        assert!(uploaded_components > 0);
        installed_bytes = weights.evidence().physical_bytes();
        let mut combined = compiled2
            .initialize_with_shared_weights(runtime.clone(), host.clone(), &weights, initial("b2"))
            .unwrap_or_else(|error| {
                let message = format!("{error:?}");
                std::mem::forget(error);
                panic!("B2 same-runtime installation bind rejected: {message}")
            });
        let ComponentInitializationReceipt::SharedWeights { evidence } =
            combined.initialization_receipt()
        else {
            panic!("B2 must attach the exact original immutable installation")
        };
        assert_eq!(evidence, weights.evidence());
        assert_eq!(counted.reads.load(Ordering::SeqCst), uploaded_components);

        // Advance just B using its genuine B1 GPU prediction. Both the future
        // batch request and independent control consume this same first result.
        let first_b = execute(&mut single, b.denoising_inputs().unwrap(), "b1.pre_b");
        let pred =
            continuations::prediction(&reference, &host, &p, layout.packed(), &plan, &b, &first_b);
        b.advance(&pred).unwrap();
        control_b.advance(&pred).unwrap();
        assert_eq!((a.evaluation(), b.evaluation()), (0, 1));

        for round in 0..2 {
            let independent = [
                execute(
                    &mut single,
                    control_a.denoising_inputs().unwrap(),
                    &format!("b1.a.{round}"),
                ),
                execute(
                    &mut single,
                    control_b.denoising_inputs().unwrap(),
                    &format!("b1.b.{round}"),
                ),
            ];
            assert_ne!(
                independent[0].bytes().bytes(),
                independent[1].bytes().bytes(),
                "distinct request values are necessary for isolation"
            );
            let control_predictions = [&control_a, &control_b]
                .into_iter()
                .zip(&independent)
                .map(|(request, output)| {
                    continuations::prediction(
                        &reference,
                        &host,
                        &p,
                        layout.packed(),
                        &plan,
                        request,
                        output,
                    )
                })
                .collect::<Vec<_>>();
            let before = [&a, &b].map(|r| {
                (
                    r.evaluation(),
                    r.video().bytes().bytes().to_vec(),
                    r.audio().bytes().bytes().to_vec(),
                )
            });
            let order = if round == 0 { [0, 1] } else { [1, 0] };
            let members = if round == 0 {
                vec![&mut a, &mut b]
            } else {
                vec![&mut b, &mut a]
            };
            let batch = batch_plan.bind(members).unwrap();
            let joined = execute(
                &mut combined,
                batch.inputs().unwrap(),
                &format!("b2.{round}"),
            );
            let stride = independent[0].bytes().len();
            assert_eq!(joined.bytes().len(), 2 * stride);
            let mut predictions = Vec::new();
            for slot in 0..2 {
                let member = order[slot];
                assert_eq!(batch.member(slot).unwrap().1, before[member].0);
                maximum_ulp = maximum_ulp.max(compare(
                    &joined.bytes().bytes()[slot * stride..(slot + 1) * stride],
                    independent[member].bytes().bytes(),
                ));
                let (head, source) = FastH3FinalOutputSource::prepare(
                    &p,
                    &plan,
                    layout.packed(),
                    before[member].0,
                    &host,
                    continuations::final_limits(),
                )
                .unwrap();
                let inputs = batch.final_output_inputs(slot, &head, &joined).unwrap();
                let hidden = inputs
                    .iter()
                    .find(|v| v.value_id().as_str() == FastH3FinalOutputFamily::INPUT_ID)
                    .unwrap();
                assert_eq!(
                    hidden.bytes().bytes(),
                    &joined.bytes().bytes()[slot * stride..(slot + 1) * stride]
                );
                predictions.push(run(
                    &reference,
                    &host,
                    FastH3FinalOutputFamily::new(continuations::final_limits()).unwrap(),
                    &head,
                    FastH3FinalOutputFamily::PROFILE_ID,
                    head.compile_options(continuations::final_limits()).unwrap(),
                    ComponentWeightSource::Stored(&source),
                    inputs,
                ));
            }
            if round == 0 {
                let usage = host.usage().unwrap().allocated_bytes();
                let bad = vec![predictions[0].clone(), vec![]];
                assert!(batch.advance(&bad).is_err());
                assert_eq!(host.usage().unwrap().allocated_bytes(), usage);
                for (request, saved) in [&a, &b].into_iter().zip(&before) {
                    assert_eq!(request.evaluation(), saved.0);
                    assert_eq!(request.video().bytes().bytes(), saved.1);
                    assert_eq!(request.audio().bytes().bytes(), saved.2);
                }
                // Hold A exactly fixed and replace only its batch partner.
                // Unlike the B1/B2 arithmetic budget, isolation within the same
                // B2 plan requires bit identity for A's entire output segment.
                let alternative = batch_plan.bind(vec![&mut a, &mut other_partner]).unwrap();
                let changed = execute(
                    &mut combined,
                    alternative.inputs().unwrap(),
                    "b2.changed_partner",
                );
                assert_eq!(
                    &changed.bytes().bytes()[..stride],
                    &joined.bytes().bytes()[..stride],
                    "request A depends on its batch partner"
                );
                assert_ne!(
                    &changed.bytes().bytes()[stride..],
                    &joined.bytes().bytes()[stride..],
                    "partner perturbation must change its own output"
                );
                drop(alternative);
                batch_plan
                    .bind(vec![&mut a, &mut b])
                    .unwrap()
                    .advance(&predictions)
                    .unwrap();
            } else {
                batch.advance(&predictions).unwrap();
            }
            control_a.advance(&control_predictions[0]).unwrap();
            control_b.advance(&control_predictions[1]).unwrap();
            assert_eq!(
                (a.evaluation(), b.evaluation()),
                (control_a.evaluation(), control_b.evaluation())
            );
            assert_eq!((a.evaluation(), b.evaluation()), (round + 1, round + 2));
            // The solver follows the outputs of its own execution. Different
            // batch shapes may round within the declared BF16 comparison budget.
            // Next comparison uses those original independent continuations.
            assert_eq!(counted.reads.load(Ordering::SeqCst), uploaded_components);
        }
        close(single);
        close(combined);
        match weights.close() {
            Ok(SharedComponentWeightsCloseOutcome::Closed(receipt)) => {
                assert!(receipt.released_static_resources() > 0)
            }
            Ok(_) => panic!("closed B1/B2 sessions still retain shared weights"),
            Err(error) => {
                let message = format!("{:?}", error.failure());
                std::mem::forget(error);
                panic!("shared installation retirement failed: {message}")
            }
        }
    }
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    eprintln!("FastH3 CUDA transient B1/B2: two original NVFP4 blocks, one runtime/native owner and shared weight installation ({uploaded_components} source reads/{installed_bytes} bytes), distinct requests eval0/1 then swap slots eval1/2, maxBF16ULP={maximum_ulp}; GPU forward -> checked partition -> Reference PDD heads -> atomic solver commit; failed second prediction leaves both unchanged; all owners retired. No full FastVideo quality or product CB claim.");
}
