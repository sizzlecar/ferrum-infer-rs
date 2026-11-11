#![cfg(feature = "cuda")]
//! Real Token Core provider qualification, separate from raw-kernel B2/F64 tests.
#[path = "support/gated_delta_candidate_component.rs"]
mod fixture;
use ferrum_runtime::*;
use fixture::*;

#[test]
#[ignore = "requires an actual CUDA device and qualified native operator lock"]
fn cuda_candidate_provider_admits_both_tails_rejects_aliases_and_retains_host_owners() {
    let composition = composition();
    // Qualify the actual closed H3 F16 profile. F32Master is not admitted by
    // this managed product factory and has no complete-provider claim here.
    for rows in [1, 2, 5] {
        let c = Config {
            rows,
            candidates: true,
            f32_hidden: false,
            log_rate: false,
            grouped: false,
        };
        let compiled = compile(c, &composition).unwrap();
        let plan = compiled.executable().execution_plan();
        let memory = plan.payload().memory();
        let required = memory.static_bytes() + memory.minimum_runnable_request_bytes();
        let ordinary = compile(
            Config {
                candidates: false,
                ..c
            },
            &composition,
        )
        .unwrap();
        let ordinary_memory = ordinary.executable().execution_plan().payload().memory();
        assert!(
            required
                >= ordinary_memory.static_bytes()
                    + ordinary_memory.minimum_runnable_request_bytes()
                    + 2 * (CONV_BYTES + DELTA_BYTES) as u64
        );
        // Both full tails are required even for N=1. Failure occurs before
        // any session is given an allocation or initialization permit.
        assert!(matches!(
            compile_with_capacity(c, &composition, required - 1),
            Err(VNextError::PlanCapacityExceeded { .. })
        ));
        let contract = gated_delta_candidate_scan_f16_contract().unwrap();
        let values = plan.payload().nodes()[0].values();
        contract
            .descriptor()
            .validate_resolved_bindings(values)
            .unwrap();
        for (destination, alias) in [(10, 0), (11, 8), (12, 9), (11, 10)] {
            let mut bad = values.to_vec();
            let output = &values[destination];
            let source = &values[alias].storage().components()[0];
            bad[destination] = ResolvedValueBinding::new(
                output.value_id().clone(),
                output.role(),
                output.ordinal(),
                output.tensor().clone(),
                output.access(),
                output.alias().clone(),
                output.usage(),
                output.weight().cloned(),
                ResolvedValueStorage::single(
                    source.resource_id().clone(),
                    source.offset_bytes(),
                    output.tensor().minimum_storage_bytes().unwrap(),
                    output.tensor().element_type(),
                )
                .unwrap(),
            )
            .unwrap();
            assert!(contract
                .descriptor()
                .validate_resolved_bindings(&bad)
                .is_err());
        }
        let account = host();
        // Isolate provider arithmetic from the independently preserved elastic
        // floor-fragmentation failure. Reserve only each compiled pool ceiling.
        let mut session = session_preprovisioned(c, &composition, &account);
        let conv = encode(
            (0..CONV_BYTES / 2).map(|i| (i as f32 * 0.19).sin() * 0.2),
            ElementType::F16,
        );
        let delta = encode(
            (0..DELTA_BYTES / 4).map(|i| (i as f32 * 0.27).cos() * 0.1),
            ElementType::F32,
        );
        let x = encode(
            (0..rows as usize * H).map(|i| (i as f32 * 0.31).sin() + 0.000123),
            c.dtype(),
        );
        // Exercise the complete ordinary provider with the same Stored
        // weights and managed inputs, including projections and residual.
        // The candidate route changes state custody, not the hidden output.
        let ordinary_config = Config {
            candidates: false,
            ..c
        };
        let ordinary_account = host();
        let mut ordinary_session =
            fixture::session_preprovisioned(ordinary_config, &composition, &ordinary_account);
        let ordinary_output = run(
            &mut ordinary_session,
            &ordinary_account,
            ordinary_config,
            &x,
            &conv,
            &delta,
        );
        let ordinary_y = bytes(&ordinary_output, Y).to_vec();
        close(ordinary_session);
        drop(ordinary_output);
        assert_eq!(ordinary_account.usage().unwrap().committed_bytes(), 0);
        let first = run(&mut session, &account, c, &x, &conv, &delta);
        assert_eq!(
            bytes(&first, Y),
            ordinary_y,
            "complete candidate and ordinary provider Y must be byte-identical: {c:?}"
        );
        let saved = first
            .terminal
            .iter()
            .map(|(id, b)| (id.clone(), b.bytes().to_vec()))
            .collect::<Vec<_>>();
        drop(first);
        let second = run(&mut session, &account, c, &x, &conv, &delta);
        assert_eq!(bytes(&second, CC).len(), 2 * CONV_BYTES);
        assert_eq!(bytes(&second, DD).len(), 2 * DELTA_BYTES);
        for word in bytes(&second, DD).chunks_exact(4) {
            assert!(f32::from_le_bytes(word.try_into().unwrap()).is_finite());
        }
        if rows == 1 {
            assert_eq!(&bytes(&second, CC)[..CONV_BYTES], conv);
            assert_eq!(&bytes(&second, DD)[..DELTA_BYTES], delta);
        }
        let snapshot = second
            .terminal
            .iter()
            .map(|(id, b)| (id.clone(), b.bytes().to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(
            snapshot, saved,
            "previous request must not mutate next request state"
        );
        close(session);
        for (_, output) in &second.terminal {
            assert!(account.owns(output));
        }
        assert!(account.usage().unwrap().allocated_bytes() >= 2 * (CONV_BYTES + DELTA_BYTES));
        assert_eq!(
            second
                .terminal
                .iter()
                .map(|(id, b)| (id.clone(), b.bytes().to_vec()))
                .collect::<Vec<_>>(),
            saved
        );
        drop(second);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}
