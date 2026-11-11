//! Actual Core token execution, not a direct call to the arithmetic helper.
mod fixture;
use crate::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_runtime::*;
use fixture::*;

const CAPACITY: u64 = 1 << 20;
fn composition(name: &str) -> CpuVNextComposition {
    CpuVNextComposition::create(id(name), CAPACITY).unwrap()
}
#[test]
fn cpu_tail_head_same_token_plan_selects_last_two_and_releases_real_owners() {
    let composition = composition("device.fixture.tail.lifecycle");
    let account = host();
    let mut session = session(7, &composition, &account);
    let plan_hash = session
        .compilation
        .executable()
        .execution_plan()
        .plan_hash()
        .to_owned();
    let rows = [
        [0., 0., 50., 0.], // deliberately large earlier score: must not win a later row.
        [1., 1., 0., 0.],  // tokens 0/1 tie only AFTER the original F16 logit rounding.
        [-1., 0., 0., 0.],
        [0., 2., 0., 0.],
        [0., 0., 3., 0.],
        [-2., 0., 0., 0.],
        [1., 1., 0., 0.],
    ];
    let input = encode(rows.into_iter().flatten());
    let mut retained = None;
    for (count, expected) in [(1, [4u32, 4]), (2, [4, 0]), (7, [2, 0]), (1, [4, 4])] {
        drop(retained.take());
        let output = run(
            &mut session,
            &account,
            count,
            &input[..count as usize * H * 2],
        );
        let actual = bytes(&output, IDS)
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "actual row count {count}");
        let left = count.saturating_sub(2) as usize;
        let right = count as usize - 1;
        let expected_hidden = [
            &input[left * H * 2..(left + 1) * H * 2],
            &input[right * H * 2..(right + 1) * H * 2],
        ]
        .concat();
        assert_eq!(
            bytes(&output, CARRIER),
            expected_hidden,
            "retain original F16 hidden rows"
        );
        assert_eq!(
            session
                .compilation
                .executable()
                .execution_plan()
                .plan_hash(),
            &plan_hash
        );
        assert!(output.terminal.iter().all(|(_, value)| account.owns(value)));
        retained = Some(output);
    }
    let output = retained.unwrap();
    let saved_ids = bytes(&output, IDS).to_vec();
    let saved_hidden = bytes(&output, CARRIER).to_vec();
    close(session);
    assert_eq!(composition.runtime().resident_bytes(), 0);
    assert!(account.usage().unwrap().allocated_bytes() > 0);
    assert_eq!(bytes(&output, IDS), saved_ids);
    assert_eq!(bytes(&output, CARRIER), saved_hidden);
    drop(output);
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn cpu_tail_head_compiled_bounds_charge_fixed_outputs_and_reject_aliases() {
    let composition = composition("device.fixture.tail.bounds");
    let short = compile(1, &composition, CAPACITY).unwrap();
    let long = compile(257, &composition, CAPACITY).unwrap();
    for compilation in [&short, &long] {
        let plan = compilation.executable().execution_plan();
        let node = &plan.payload().nodes()[0];
        let values = node.values();
        let outputs = values
            .iter()
            .filter(|v| v.role() == ResolvedValueRole::Output)
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].tensor().dimensions(), &[2]);
        assert_eq!(outputs[0].tensor().minimum_storage_bytes().unwrap(), 8);
        assert_eq!(outputs[1].tensor().dimensions(), &[2, H as u64]);
        assert_eq!(
            outputs[1].tensor().minimum_storage_bytes().unwrap(),
            (2 * H * 2) as u64
        );
        // One vocabulary row per actual participant, independent of its
        // token count; the public requirement includes physical alignment.
        let scratch = node.provider_resources().scratch().unwrap();
        let alignment = scratch.alignment_bytes();
        let expected = ((V * 2) as u64).div_ceil(alignment) * alignment;
        for rows in [1, 257] {
            let tokens = vec![1_u32; rows];
            let work =
                ResourceWorkShape::single(TokenSpanWork::from_token_ids(&tokens, 0..rows).unwrap())
                    .unwrap();
            assert_eq!(scratch.evaluate_bytes(&work).unwrap(), expected);
        }
        tail_two_greedy_f16_contract()
            .unwrap()
            .descriptor()
            .validate_resolved_bindings(values)
            .unwrap();
    }
    let plan = long.executable().execution_plan();
    let required = plan.payload().memory().static_bytes()
        + plan.payload().memory().minimum_runnable_request_bytes();
    assert!(matches!(
        compile(257, &composition, required - 1),
        Err(VNextError::PlanCapacityExceeded { .. })
    ));
    assert_eq!(
        composition.runtime().resident_bytes(),
        0,
        "metadata failure must not allocate"
    );
    let values = plan.payload().nodes()[0].values();
    let contract = tail_two_greedy_f16_contract().unwrap();
    for (destination, source) in [(2, 0), (2, 1), (2, 3), (3, 0), (3, 1), (3, 2)] {
        let output = &values[destination];
        let source = &values[source].storage().components()[0];
        let mut aliased = values.to_vec();
        aliased[destination] = ResolvedValueBinding::new(
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
        assert!(
            contract
                .descriptor()
                .validate_resolved_bindings(&aliased)
                .is_err(),
            "every output must remain disjoint from inputs and its sibling"
        );
    }
}
