//! One counter per actual model lookup result, including unknown results.
//! Labels are finite protocol enums; no request, family, shape or model ID is
//! retained. These counters do not train, publish, or authorize execution.
use ferrum_interfaces::execution_cost::StatisticalEvidenceUnknown as Evidence;
use ferrum_scheduler::implementations::continuous::cost_model::statistical::model::ModelUnknown;
use ferrum_scheduler::implementations::continuous::cost_model::structured::StructuredUnknown;

pub(super) fn record_structured<T>(result: &Result<T, StructuredUnknown>) {
    let (status, reason) = match result {
        Ok(_) => ("known", "none"),
        Err(reason) => ("unknown", structured_label(*reason)),
    };
    metrics::counter!("ferrum.engine.structured_cost_queries_total",
        "scope" => "candidate", "result" => status, "reason" => reason)
    .increment(1);
}

pub(super) fn record_structured_v2<T>(result: &Result<T, StructuredUnknown>) {
    let (status, reason) = match result {
        Ok(_) => ("known", "none"),
        Err(reason) => ("unknown", structured_label(*reason)),
    };
    metrics::counter!("ferrum.engine.structured_v2_cost_queries_total","scope"=>"candidate","result"=>status,"reason"=>reason).increment(1);
}

fn structured_label(reason: StructuredUnknown) -> &'static str {
    match reason {
        StructuredUnknown::MissingEvidence => "missing_evidence",
        StructuredUnknown::UnsupportedScope => "unsupported_scope",
        StructuredUnknown::InvalidInput => "invalid_input",
        StructuredUnknown::InvalidSettings => "invalid_settings",
        StructuredUnknown::WrongDomain => "wrong_domain",
        StructuredUnknown::WrongSource => "wrong_source",
        StructuredUnknown::WrongProtocol => "wrong_protocol",
        StructuredUnknown::WrongFingerprint => "wrong_fingerprint",
        StructuredUnknown::PhaseLeakage => "phase_leakage",
        StructuredUnknown::DuplicateRecord => "duplicate_record",
        StructuredUnknown::InvalidSample => "invalid_sample",
        StructuredUnknown::Capacity => "capacity",
        StructuredUnknown::Clock => "clock",
        StructuredUnknown::Stale => "stale",
        StructuredUnknown::InsufficientSamples => "insufficient_samples",
        StructuredUnknown::InsufficientRedundancy => "insufficient_redundancy",
        StructuredUnknown::IncompletePhasePopulation => "incomplete_phase_population",
        StructuredUnknown::JointSupport => "joint_support",
        StructuredUnknown::UnidentifiedDirection => "unidentified_direction",
        StructuredUnknown::IllConditioned => "ill_conditioned",
        StructuredUnknown::Numerical => "numerical",
        StructuredUnknown::QualificationCoverage => "qualification_coverage",
        StructuredUnknown::QualificationUnderestimate => "qualification_underestimate",
    }
}

pub(super) enum QueryScope {
    Candidate,
    RetrospectiveActual,
}

pub(super) fn record<T>(scope: QueryScope, result: &Result<T, ModelUnknown>) {
    let scope = match scope {
        QueryScope::Candidate => "candidate",
        QueryScope::RetrospectiveActual => "retrospective_actual",
    };
    let (status, reason) = match result {
        Ok(_) => ("known", "none"),
        Err(reason) => ("unknown", label(*reason)),
    };
    metrics::counter!("ferrum.engine.selected_cost_queries_total",
        "scope" => scope, "result" => status, "reason" => reason)
    .increment(1);
}

fn label(reason: ModelUnknown) -> &'static str {
    match reason {
        ModelUnknown::Evidence(reason) => match reason {
            Evidence::MissingProducer => "evidence_missing_producer",
            Evidence::InvalidAlgorithm => "evidence_invalid_algorithm",
            Evidence::InvalidWork => "evidence_invalid_work",
            Evidence::CommandMismatch => "evidence_command_mismatch",
            Evidence::ExactBindingMismatch => "evidence_exact_binding_mismatch",
            Evidence::MissingHostDomain => "evidence_missing_host_domain",
            Evidence::UnsupportedWave => "evidence_unsupported_wave",
            Evidence::UnsupportedReplay => "evidence_unsupported_replay",
            Evidence::Capacity => "evidence_capacity",
            Evidence::Overflow => "evidence_overflow",
        },
        ModelUnknown::InvalidSettings => "invalid_settings",
        ModelUnknown::InvalidSample => "invalid_sample",
        ModelUnknown::WrongFingerprint => "wrong_fingerprint",
        ModelUnknown::WrongSource => "wrong_source",
        ModelUnknown::PhaseLeakage => "phase_leakage",
        ModelUnknown::DuplicateRecord => "duplicate_record",
        ModelUnknown::Capacity => "capacity",
        ModelUnknown::Clock => "clock",
        ModelUnknown::Stale => "stale",
        ModelUnknown::InsufficientFit => "insufficient_fit",
        ModelUnknown::InsufficientResidual => "insufficient_residual",
        ModelUnknown::FamilyMissing => "family_missing",
        ModelUnknown::JointSupport => "joint_support",
        ModelUnknown::Numerical => "numerical",
        ModelUnknown::RuntimeValidity => "runtime_validity",
    }
}
