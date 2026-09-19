//! Evidence acquisition may recover from transport failure, but never from a
//! successfully observed invalid origin, incomplete inventory or failed check.
use super::*;

fn execution(run_id: &str, job_id: u64, conclusion: &str, completed_at: &str) -> Execution {
    Execution {
        run_id: run_id.into(),
        job_id,
        status: "completed".into(),
        conclusion: conclusion.into(),
        completed_at: completed_at.into(),
    }
}

fn successful_execution() -> Execution {
    execution("123", 456, "success", "2026-09-19T07:00:00Z")
}

#[test]
fn failed_network_read_can_recover_with_one_complete_inventory() {
    let complete = "total\t1\nitem\t456\tsuccess\n";
    let mut responses = [
        Err("connection reset during paginated Actions read".to_owned()),
        Ok(complete.to_owned()),
    ]
    .into_iter();
    let result = retry_read(
        "read Actions jobs",
        || responses.next().expect("do not retry a successful read"),
        |_| {},
    )
    .unwrap();
    assert_eq!(result, complete);
    assert_eq!(inventory_rows(&result, 0).unwrap(), ["456\tsuccess"]);
}

#[test]
fn exhausted_acquisition_preserves_the_cause_and_never_authorizes_reuse() {
    let unavailable: Result<Checks, String> = retry_read(
        "download prior CI origin",
        || Err("artifact endpoint connection reset".to_owned()),
        |_| {},
    );
    let error = unavailable.as_ref().unwrap_err();
    assert!(error.contains("download prior CI origin"), "{error}");
    assert!(
        error.contains("artifact endpoint connection reset"),
        "{error}"
    );

    let mut plan = Plan::fresh(Checks::ALL);
    // Failure to establish CUDA evidence must preserve independently proved
    // checks while withholding only the requested reuse authorization.
    plan.reused = Checks(Check::Metal.bit());
    let original = plan;
    let mut notes = Vec::new();
    record_reuse(
        &mut plan,
        &mut notes,
        Check::Cuda,
        &successful_execution(),
        &unavailable,
    );
    assert_eq!(plan, original);
    let diagnostic = notes.join("\n");
    assert!(diagnostic.contains(Check::Cuda.name()), "{diagnostic}");
    assert!(diagnostic.contains("123"), "{diagnostic}");
    assert!(
        diagnostic.contains("artifact endpoint connection reset"),
        "{diagnostic}"
    );
}

#[test]
fn valid_transport_does_not_retry_invalid_origin_or_incomplete_inventory() {
    let origin = format!("revision={}\nrun_id=999\n", "a".repeat(40));
    let inventory = "total\t2\nitem\t456\tsuccess\n".to_owned();
    for (description, invalid) in [("origin", origin), ("inventory", inventory)] {
        let mut response = Some(invalid);
        let payload = retry_read(
            description,
            || {
                Ok(response
                    .take()
                    .expect("valid transport must not be retried to replace bad evidence"))
            },
            |_| panic!("valid transport needs no retry delay"),
        )
        .unwrap();
        match description {
            "origin" => assert!(parse_origin(&payload, "123").is_err()),
            "inventory" => assert!(inventory_rows(&payload, 0).is_err()),
            _ => unreachable!(),
        }
    }
}

#[test]
fn successful_source_requires_unchanged_scope_for_the_selected_check() {
    let source = successful_execution();
    for (changed, expected_reuse) in [
        (Checks::default(), true),
        (Checks(Check::Metal.bit()), true),
        (Checks(Check::Cuda.bit()), false),
    ] {
        let mut plan = Plan::fresh(Checks::ALL);
        let mut notes = Vec::new();
        record_reuse(&mut plan, &mut notes, Check::Cuda, &source, &Ok(changed));
        assert_eq!(plan.reused.has(Check::Cuda), expected_reuse);
        assert_eq!(plan.required, Checks::ALL);
        assert_eq!(
            plan.reused.0 & !Check::Cuda.bit(),
            0,
            "one source check must not authorize other checks"
        );
    }
}

#[test]
fn newer_failed_execution_blocks_an_older_success_after_successful_acquisition() {
    let executions = [
        successful_execution(),
        execution("124", 457, "failure", "2026-09-19T07:01:00Z"),
    ];
    let selected = latest(&executions).unwrap().unwrap();
    assert_eq!(selected.conclusion, "failure");
    let unchanged = retry_read("compare source scope", || Ok(Checks::default()), |_| {});
    let mut plan = Plan::fresh(Checks::ALL);
    let original = plan;
    let mut notes = Vec::new();
    record_reuse(&mut plan, &mut notes, Check::Cuda, selected, &unchanged);
    assert_eq!(plan, original);
}
