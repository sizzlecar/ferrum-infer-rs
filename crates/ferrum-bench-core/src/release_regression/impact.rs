//! Conservative component mapping for a complete PR or release diff.
//! Only the declared backend module boundaries narrow accelerator reach. Shared
//! and unfamiliar paths retain component-wide scope; model identity is not inferred.
use super::types::{Backend, ChangeArea, Impact, PathImpact};
use std::collections::BTreeSet;

const ALL_AREAS: &[ChangeArea] = &ChangeArea::ALL;

/// Analyze the union of changed repository-relative paths. Callers must supply
/// the complete diff, including removed/renamed-away paths, and handle git errors.
/// Empty input means no changed paths; it is not a claim of release coverage.
/// Unknown/malformed paths retain their identity and expand to every area.
/// Output is deterministic and independent of input order or duplicate paths.
pub fn analyze_paths<I, S>(paths: I) -> Impact
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let paths: BTreeSet<String> = paths
        .into_iter()
        .map(|path| path.as_ref().to_owned())
        .collect();
    let mut result = Impact {
        areas: Vec::new(),
        paths: Vec::new(),
        unknown_paths: Vec::new(),
        product_contract_changed: false,
    };
    for path in paths {
        let classification = classify(&path);
        let (areas, reason) = match classification {
            Some(value) => value,
            None => {
                result.unknown_paths.push(path.clone());
                (
                    ALL_AREAS.to_vec(),
                    "unmapped path: retain full component scope and require review",
                )
            }
        };
        if matches!(path.as_str(), "README.md" | "README_zh.md") {
            result.product_contract_changed = true;
        }
        for area in &areas {
            if !result.areas.contains(area) {
                result.areas.push(*area);
            }
        }
        result.paths.push(PathImpact {
            path,
            areas,
            reason: reason.to_owned(),
            execution_paths: None,
        });
    }
    // Use catalogue order without requiring an incidental enum discriminant order.
    result.areas = ALL_AREAS
        .iter()
        .copied()
        .filter(|area| result.areas.contains(area))
        .collect();
    result
}

/// These are the actual isolated modules declared by backend/mod.rs. None means
/// shared or unreviewed reach, never that no backend is affected. Keep this proof
/// attached to each contributing path; an unrelated Metal path cannot narrow a
/// shared protocol, dependency or model-program change in the same diff.
pub(super) fn path_backend(path: &str) -> Option<Backend> {
    if !valid_relative_path(path) {
        return None;
    }
    let relative = path.strip_prefix("crates/ferrum-kernels/src/backend/")?;
    if relative == "cpu.rs" {
        return Some(Backend::Cpu);
    }
    for (directory, backend) in [("metal/", Backend::Metal), ("cuda/", Backend::Cuda)] {
        if relative
            .strip_prefix(directory)
            .is_some_and(|file| !file.is_empty())
        {
            return Some(backend);
        }
    }
    None
}

fn classify(path: &str) -> Option<(Vec<ChangeArea>, &'static str)> {
    use ChangeArea::*;
    if !valid_relative_path(path) {
        return None;
    }
    if matches!(path, "README.md" | "README_zh.md") {
        return Some((
            Vec::new(),
            "README product promises must be reviewed independently of code impact",
        ));
    }
    if (path.starts_with("docs/") && path.ends_with(".md"))
        || matches!(
            path,
            "CHANGELOG.md" | "CONTRIBUTING.md" | "LICENSE" | "LICENSE-MIT" | "LICENSE-APACHE"
        )
    {
        return Some((
            Vec::new(),
            "documentation-only path; no production component inferred",
        ));
    }
    if path == "docs/release-regression-catalog.json" {
        return Some((
            vec![Validation],
            "declared regression inventory and coverage rules",
        ));
    }
    if matches!(
        path,
        "Cargo.toml" | "Cargo.lock" | "rust-toolchain.toml" | "rust-toolchain"
    ) || path.starts_with(".cargo/")
    {
        return Some((
            ALL_AREAS.to_vec(),
            "shared dependency/toolchain change: conservatively include all components",
        ));
    }
    if path == ".gitignore" {
        return Some((
            vec![Build],
            "source packaging inputs; retain build and installation checks",
        ));
    }
    if path.starts_with(".github/workflows/")
        || path.starts_with(".github/ci/")
        || matches!(
            path,
            ".github/actionlint.yaml"
                | ".github/actionlint.yml"
                | ".github/release-performance.json"
        )
    {
        return Some((
            vec![Build, Validation],
            "build/release execution and required-check selection",
        ));
    }
    if path == "AGENTS.md" {
        return Some((vec![Validation], "repository validation requirements"));
    }
    if path.starts_with("tests/") || path.starts_with("benches/") {
        return Some((vec![Validation], "shared validation inputs and checks"));
    }
    let rest = path.strip_prefix("crates/")?;
    let (component, relative) = rest.split_once('/')?;
    if component == "ferrum-devtools" {
        if relative == "Cargo.toml" || (relative.starts_with("src/") && relative.ends_with(".rs")) {
            return Some((vec![Build, Validation],
                "private development-tool crate; manifest changes require complete snapshot validation of publish=false and dependency isolation"));
        }
        // Build hooks, assets and additional configuration are not inferred from
        // this private crate's name. Its reviewed Rust source boundary is explicit.
        return None;
    }
    if matches!(relative, "Cargo.toml" | "build.rs") {
        // A crate dependency or build hook can change behavior beyond its own source.
        return Some((
            ALL_AREAS.to_vec(),
            "crate build/dependency change: conservatively include all components",
        ));
    }
    // continuous_engine.rs declares this exact module under cfg(test). A
    // changed production parent/import still contributes its own full reach.
    if component == "ferrum-engine" && relative == "src/continuous_engine/tests.rs" {
        return Some((
            vec![Validation],
            "engine unit-test module declared only under cfg(test)",
        ));
    }
    // Cargo's top-level tests/*.rs files are independent integration targets.
    // Keep nested fixtures/helpers and unknown crates conservative: their use
    // cannot be established from a directory name alone. A production import
    // added alongside a test still contributes its own production source diff.
    if known_component(component)
        && relative.strip_prefix("tests/").is_some_and(|name| {
            !name.contains('/')
                && name
                    .strip_suffix(".rs")
                    .is_some_and(|stem| !stem.is_empty())
        })
    {
        return Some((
            vec![Validation],
            "known crate's top-level Cargo integration-test target",
        ));
    }
    // These modules are only connected to the independent staged-binary test
    // executable; do not infer the same for other CLI examples or nested helpers.
    if component == "ferrum-cli"
        && matches!(
            relative,
            "examples/model_regression.rs"
                | "examples/model_regression/cases.rs"
                | "examples/model_regression/process.rs"
                | "examples/model_regression/protocol.rs"
                | "examples/model_regression/stop.rs"
                | "examples/model_regression/stop_tests.rs"
                | "examples/model_regression/identity.rs"
                | "examples/model_regression/identity_tests.rs"
                | "examples/model_regression/boundaries.rs"
                | "examples/model_regression/boundary_tests.rs"
        )
    {
        return Some((
            vec![Validation],
            "independent staged-binary regression runner and its declared helper modules",
        ));
    }
    if component == "ferrum-server"
        && matches!(
            relative,
            "src/axum_server/tests/engine_stop_contract.rs"
                | "src/axum_server/tests/engine_stop_contract/executor.rs"
                | "src/axum_server/tests/engine_stop_contract/structured.rs"
                | "src/axum_server/tests/engine_stop_contract/tools.rs"
        )
    {
        return Some((vec![Validation],
            "explicit production-engine protocol fixture modules reachable only through axum_server's cfg(test) tests module"));
    }
    if component == "ferrum-cli" && relative == "src/source_resolver.rs" {
        return Some((vec![Download, Template, Scheduler, Kv],
            "source/metadata/template selection and runtime presets include paged KV routing; retain loading, protocol and resource obligations without inferring changed operators"));
    }
    if component == "ferrum-server" && relative == "src/axum_server.rs" {
        return Some((vec![Template, Termination, Structured, Tools, Scheduler],
            "HTTP request/output adaptation, conversation history and admission/cancellation; message-history caches do not implement device KV save/restore"));
    }
    if component == "ferrum-models"
        && matches!(
            relative,
            "src/hf_download.rs" | "src/hf_download/selection.rs"
        )
    {
        return Some((
            vec![Download],
            "Hub transfer, immutable source selection, shard closure and cache publication",
        ));
    }
    if (component == "ferrum-models" && relative == "src/hf_download/download_tests.rs")
        || (component == "ferrum-cli" && relative == "tests/download_jsonl/hub.rs")
    {
        return Some((
            vec![Validation],
            "explicit local Hub test fixture imported only by its registered regression target",
        ));
    }
    if component == "ferrum-types"
        && matches!(
            relative,
            "src/reasoning.rs" | "src/reasoning/gemma.rs" | "src/harmony.rs"
        )
    {
        return Some((vec![Template, Termination, Structured, Tools],
            "declared reasoning/message framing and terminal interpretation feed run and HTTP output/history adapters"));
    }
    if component == "ferrum-kernels" && path_backend(path).is_some() {
        return Some((
            vec![Kernel],
            "isolated backend implementation: retain operator numerics, boundaries, model forward and performance; model architecture/state is a separate responsibility",
        ));
    }
    if component == "ferrum-bench-core" {
        return classify_bench_core(relative);
    }
    let (areas, reason) = match component {
        "ferrum-types" | "ferrum-interfaces" => (
            ALL_AREAS.to_vec(),
            "shared contracts feed loading, request conversion, execution and validation",
        ),
        "ferrum-engine" => (
            vec![
                Template,
                Termination,
                Structured,
                Tools,
                Scheduler,
                Kv,
                Kernel,
                Architecture,
            ],
            "shared execution: retain protocol, sampling, resource and backend interactions",
        ),
        "ferrum-sampler" => (
            vec![Termination, Structured, Tools],
            "sampling, grammar and terminal selection interact across output protocols",
        ),
        "ferrum-tokenizer" => (
            vec![Template, Termination, Structured, Tools],
            "token identity and decoding affect templates, markers and constraints",
        ),
        "ferrum-scheduler" => (
            vec![Scheduler, Kv, Architecture],
            "admission and batching affect resource ownership and architecture state",
        ),
        "ferrum-kv" => (
            vec![Scheduler, Kv, Architecture],
            "cache state, capacity and reuse interact with scheduled architecture execution",
        ),
        "ferrum-kernels" => (
            vec![Kernel, Architecture],
            "shared or unreviewed kernel path: retain full backend and architecture scope",
        ),
        "ferrum-quantization" => (
            vec![Download, Kernel, Architecture],
            "quantized source representation and actual numerical execution",
        ),
        "ferrum-native-ops" | "ferrum-native-ops-builder" => (
            vec![Build, Kernel, Architecture],
            "native build/runtime dependencies and backend execution paths",
        ),
        "ferrum-models" => (
            vec![Download, Template, Termination, Kv, Kernel, Architecture],
            "model sources, configuration and execution; retain component-wide reach",
        ),
        "ferrum-server" => (
            vec![Template, Termination, Structured, Tools, Scheduler, Kv],
            "HTTP conversion and adapters interact with execution and request lifetime",
        ),
        "ferrum-cli" => (
            ALL_AREAS.to_vec(),
            "product entrypoints share source resolution, configuration and execution",
        ),
        "ferrum-testkit" => (
            vec![Validation],
            "validation oracles, workload selection and measurement infrastructure",
        ),
        _ => return None,
    };
    Some((areas, reason))
}

fn known_component(component: &str) -> bool {
    matches!(
        component,
        "ferrum-types"
            | "ferrum-interfaces"
            | "ferrum-engine"
            | "ferrum-sampler"
            | "ferrum-tokenizer"
            | "ferrum-scheduler"
            | "ferrum-kv"
            | "ferrum-kernels"
            | "ferrum-quantization"
            | "ferrum-native-ops"
            | "ferrum-native-ops-builder"
            | "ferrum-models"
            | "ferrum-server"
            | "ferrum-cli"
            | "ferrum-testkit"
            | "ferrum-bench-core"
    )
}

fn classify_bench_core(relative: &str) -> Option<(Vec<ChangeArea>, &'static str)> {
    use ChangeArea::*;
    if matches!(
        relative,
        "src/lib.rs" | "src/jsonl_journal.rs" | "src/profile.rs" | "src/trace.rs"
    ) {
        // lib.rs parses headers on the real Chat endpoint. The journal/profile/
        // trace modules are called by engine lifecycle and backend execution,
        // as well as benchmark producers; crate naming cannot erase that reach.
        return Some((
            vec![Observability, Validation],
            "shared production request metadata, profile sinks and runtime journals",
        ));
    }
    if matches!(
        relative,
        "src/arrivals.rs"
            | "src/decode_isolation.rs"
            | "src/env.rs"
            | "src/report.rs"
            | "src/stats.rs"
            | "examples/model_gate.rs"
            | "examples/model_gate/tests.rs"
            | "examples/regression_plan.rs"
            | "examples/regression_plan/scope.rs"
            | "examples/release_candidate.rs"
            | "examples/contract_checks.rs"
            | "examples/release_delivery.rs"
            | "tests/release_staging_workflows.rs"
    ) || relative.starts_with("src/release_regression/")
        || relative.starts_with("src/release_candidate/")
        || relative.starts_with("examples/release_candidate/")
        || relative.starts_with("examples/release_delivery/")
    {
        return Some((
            vec![Validation],
            "declared benchmark measurement, regression planning and candidate preparation tools",
        ));
    }
    // New modules need their callers reviewed. They may enter production just
    // as the existing journal and profile modules do.
    None
}

fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains(['\\', '\0', '\n', '\r'])
        && !path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isolated_engine_tests_and_packaging_do_not_erase_parent_execution_changes() {
        let tests = "crates/ferrum-engine/src/continuous_engine/tests.rs";
        assert_eq!(analyze_paths([tests]).areas, [ChangeArea::Validation]);
        assert_eq!(analyze_paths([".gitignore"]).areas, [ChangeArea::Build]);
        for parent in [
            "crates/ferrum-engine/src/continuous_engine.rs",
            "crates/ferrum-engine/src/continuous_engine/sequence.rs",
            "crates/ferrum-engine/src/continuous_engine/tests/unreviewed.rs",
        ] {
            assert!(analyze_paths([tests, ".gitignore", parent])
                .areas
                .contains(&ChangeArea::Kernel));
        }
    }

    #[test]
    fn shared_contracts_and_engine_expand_interacting_behaviors() {
        for path in [
            "crates/ferrum-types/src/sampling.rs",
            "crates/ferrum-engine/src/continuous_engine/sequence.rs",
        ] {
            let impact = analyze_paths([path]);
            for area in [
                ChangeArea::Template,
                ChangeArea::Termination,
                ChangeArea::Structured,
                ChangeArea::Tools,
                ChangeArea::Scheduler,
                ChangeArea::Kv,
                ChangeArea::Architecture,
            ] {
                assert!(impact.areas.contains(&area), "{path} omitted {area:?}");
            }
            assert!(impact.unknown_paths.is_empty());
        }
    }

    #[test]
    fn unknown_or_malformed_paths_are_retained_and_never_skipped() {
        for path in [
            "new-runtime/forward.rs",
            "crates/new-executor/src/lib.rs",
            "docs/fixture.json",
            "docs/../crates/ferrum-engine/src/lib.rs",
            "/tmp/input.rs",
            "",
            "crates\\ferrum-engine\\src\\lib.rs",
        ] {
            let impact = analyze_paths([path]);
            assert_eq!(impact.unknown_paths, [path]);
            assert_eq!(impact.paths[0].path, path);
            for area in ALL_AREAS {
                assert!(impact.areas.contains(area), "{path:?} omitted {area:?}");
            }
        }
    }

    #[test]
    fn readme_changes_keep_product_contract_obligations_separate() {
        let docs = analyze_paths(["docs/guide.md", "CHANGELOG.md"]);
        assert!(docs.areas.is_empty());
        assert!(!docs.product_contract_changed);
        let readme = analyze_paths(["README.md", "README_zh.md"]);
        assert!(readme.product_contract_changed);
        assert!(readme.areas.is_empty());
        assert!(readme.unknown_paths.is_empty());
    }

    #[test]
    fn full_diff_union_keeps_removed_code_and_ignores_input_order() {
        let before = "crates/ferrum-sampler/src/stop.rs";
        let after = "docs/stop.md";
        let first = analyze_paths([before, after, before]);
        let second = analyze_paths([after, before]);
        assert!(first.areas.contains(&ChangeArea::Termination));
        assert!(first.areas.contains(&ChangeArea::Structured));
        assert_eq!(first.areas, second.areas);
        let paths = |impact: &Impact| {
            impact
                .paths
                .iter()
                .map(|entry| entry.path.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(paths(&first), paths(&second));
    }

    #[test]
    fn declared_backend_modules_have_kernel_responsibility_without_model_state() {
        for (path, backend) in [
            (
                "crates/ferrum-kernels/src/backend/metal/mod.rs",
                Backend::Metal,
            ),
            (
                "crates/ferrum-kernels/src/backend/cuda/fused_silu_mul.rs",
                Backend::Cuda,
            ),
            ("crates/ferrum-kernels/src/backend/cpu.rs", Backend::Cpu),
        ] {
            assert_eq!(path_backend(path), Some(backend));
            let impact = analyze_paths([path]);
            assert_eq!(impact.areas, [ChangeArea::Kernel]);
            assert!(
                !impact.areas.contains(&ChangeArea::BackendSubmission),
                "path names do not prove submission-only changes"
            );
            assert!(impact.unknown_paths.is_empty());
        }
        for path in [
            "crates/ferrum-kernels/src/backend/traits.rs",
            "crates/ferrum-kernels/src/backend/another/ops.rs",
            "crates/ferrum-kernels/src/metal/ops.rs",
            "crates/ferrum-kernels/src/backend/metal/../cuda/ops.rs",
            "crates/ferrum-models/src/vnext/qwen35.rs",
        ] {
            assert_eq!(path_backend(path), None, "{path}");
            assert!(analyze_paths([path])
                .areas
                .contains(&ChangeArea::Architecture));
        }
    }

    #[test]
    fn backend_and_build_changes_cannot_become_protocol_only_plans() {
        for path in [
            "crates/ferrum-kernels/src/cuda/ops.rs",
            "crates/ferrum-native-ops/src/lib.rs",
            "Cargo.lock",
        ] {
            let impact = analyze_paths([path]);
            assert!(impact.areas.contains(&ChangeArea::Kernel));
            assert!(impact.areas.contains(&ChangeArea::Architecture));
        }
        let validation = analyze_paths([
            ".github/workflows/ci.yml",
            ".github/workflows/prepare-release.yml",
            ".github/workflows/release-delivery.yml",
            ".github/workflows/release-cloud-reaper.yml",
            ".github/actionlint.yaml",
            "crates/ferrum-bench-core/src/release_regression/selection.rs",
        ]);
        assert!(validation.areas.contains(&ChangeArea::Validation));
        assert!(validation.areas.contains(&ChangeArea::Build));
        assert!(validation.unknown_paths.is_empty());
        assert!(!validation.areas.contains(&ChangeArea::Kernel));
    }

    #[test]
    fn production_headers_and_sinks_retain_observability_without_inferred_numerics() {
        for path in [
            "crates/ferrum-bench-core/src/lib.rs",
            "crates/ferrum-bench-core/src/jsonl_journal.rs",
            "crates/ferrum-bench-core/src/profile.rs",
            "crates/ferrum-bench-core/src/trace.rs",
        ] {
            let impact = analyze_paths([path]);
            assert!(impact.unknown_paths.is_empty());
            for area in ALL_AREAS {
                assert_eq!(
                    impact.areas.contains(area),
                    matches!(area, ChangeArea::Observability | ChangeArea::Validation),
                    "unexpected production observability scope for {path}: {area:?}"
                );
            }
        }
    }

    #[test]
    fn known_regression_tools_do_not_inherit_inference_component_scope() {
        for path in [
            "crates/ferrum-cli/examples/model_regression.rs",
            "crates/ferrum-cli/examples/model_regression/cases.rs",
            "crates/ferrum-cli/examples/model_regression/process.rs",
            "crates/ferrum-cli/examples/model_regression/protocol.rs",
            "crates/ferrum-cli/examples/model_regression/stop.rs",
            "crates/ferrum-cli/examples/model_regression/stop_tests.rs",
            "crates/ferrum-bench-core/src/release_regression/selection.rs",
            "crates/ferrum-bench-core/src/release_candidate/staging.rs",
            "crates/ferrum-bench-core/examples/release_delivery.rs",
            "crates/ferrum-bench-core/examples/release_delivery/cloud.rs",
            "crates/ferrum-bench-core/examples/contract_checks.rs",
            "crates/ferrum-bench-core/examples/regression_plan.rs",
            "crates/ferrum-bench-core/examples/release_candidate/workspace.rs",
            "crates/ferrum-bench-core/src/stats.rs",
        ] {
            let impact = analyze_paths([path]);
            assert_eq!(impact.areas, [ChangeArea::Validation], "{path}");
            assert!(impact.unknown_paths.is_empty());
        }
        let combined = analyze_paths([
            "crates/ferrum-cli/examples/model_regression/protocol.rs",
            "crates/ferrum-bench-core/src/jsonl_journal.rs",
        ]);
        assert!(combined.areas.contains(&ChangeArea::Observability));
        assert!(combined.areas.contains(&ChangeArea::Validation));
        assert!(!combined.areas.contains(&ChangeArea::Kernel));
    }

    #[test]
    fn reviewed_engine_protocol_fixtures_are_validation_while_unknown_helpers_remain_conservative()
    {
        for path in [
            "crates/ferrum-server/src/axum_server/tests/engine_stop_contract.rs",
            "crates/ferrum-server/src/axum_server/tests/engine_stop_contract/executor.rs",
            "crates/ferrum-server/src/axum_server/tests/engine_stop_contract/structured.rs",
            "crates/ferrum-server/src/axum_server/tests/engine_stop_contract/tools.rs",
        ] {
            assert_eq!(analyze_paths([path]).areas, [ChangeArea::Validation]);
        }
        for path in [
            "crates/ferrum-server/src/axum_server/tests/engine_stop_contract/unreviewed.rs",
            "crates/ferrum-kv/src/managers/paged.rs",
        ] {
            assert!(
                analyze_paths([path]).areas.contains(&ChangeArea::Kv),
                "{path}"
            );
        }
    }

    #[test]
    fn private_devtools_source_and_manifest_keep_build_validation_but_unknown_config_does_not() {
        for path in [
            "crates/ferrum-devtools/Cargo.toml",
            "crates/ferrum-devtools/src/bin/release_delivery.rs",
            "crates/ferrum-devtools/src/bin/contract_checks.rs",
            "crates/ferrum-devtools/src/bin/release_delivery/cloud/api.rs",
        ] {
            let impact = analyze_paths([path]);
            assert_eq!(impact.areas, [ChangeArea::Build, ChangeArea::Validation]);
            assert!(impact.unknown_paths.is_empty());
        }
        for path in [
            "crates/ferrum-devtools/build.rs",
            "crates/ferrum-devtools/runtime-config.toml",
            "crates/ferrum-devtools/src/runtime-config.json",
            "crates/new-devtools/src/bin/tool.rs",
        ] {
            let impact = analyze_paths([path]);
            assert_eq!(impact.areas, ALL_AREAS);
            assert_eq!(impact.unknown_paths, [path]);
        }
        // Old release intervals still contain the removed example paths.
        for path in [
            "crates/ferrum-bench-core/examples/release_delivery.rs",
            "crates/ferrum-bench-core/examples/contract_checks.rs",
        ] {
            assert_eq!(analyze_paths([path]).areas, [ChangeArea::Validation]);
        }
    }

    #[test]
    fn unreviewed_modules_and_other_examples_are_not_assumed_validation_only() {
        for path in [
            "crates/ferrum-bench-core/src/new_runtime_sink.rs",
            "crates/ferrum-bench-core/examples/new_runner.rs",
            "crates/ferrum-bench-core/tests/shared/new_fixture.rs",
        ] {
            let impact = analyze_paths([path]);
            assert_eq!(impact.unknown_paths, [path]);
            assert_eq!(impact.areas, ALL_AREAS);
        }
        for path in [
            "crates/ferrum-cli/src/commands/run.rs",
            "crates/ferrum-cli/examples/new_runtime.rs",
            "crates/ferrum-cli/examples/model_regression/new_helper.rs",
            "crates/ferrum-cli/tests/shared/new_fixture.rs",
        ] {
            assert_eq!(analyze_paths([path]).areas, ALL_AREAS, "{path}");
        }
    }

    #[test]
    fn independent_component_changes_are_unioned_not_overwritten() {
        let impact = analyze_paths([
            "crates/ferrum-scheduler/src/lib.rs",
            "crates/ferrum-tokenizer/src/lib.rs",
        ]);
        for area in [
            ChangeArea::Scheduler,
            ChangeArea::Kv,
            ChangeArea::Template,
            ChangeArea::Termination,
            ChangeArea::Structured,
            ChangeArea::Tools,
        ] {
            assert!(impact.areas.contains(&area));
        }
        assert!(!impact.areas.contains(&ChangeArea::Download));
    }
    #[test]
    fn integration_targets_are_validation_but_nested_unknown_and_mixed_changes_are_not() {
        for component in [
            "ferrum-cli",
            "ferrum-engine",
            "ferrum-models",
            "ferrum-kernels",
            "ferrum-bench-core",
        ] {
            let test = format!("crates/{component}/tests/a_new_boundary.rs");
            assert_eq!(
                analyze_paths([test.as_str()]).areas,
                [ChangeArea::Validation]
            );
            let nested = format!("crates/{component}/tests/support/a_boundary.rs");
            assert_ne!(
                analyze_paths([nested.as_str()]).areas,
                [ChangeArea::Validation]
            );
        }
        let unknown = analyze_paths(["crates/new-component/tests/looks_like_a_test.rs"]);
        assert_eq!(unknown.areas, ALL_AREAS);
        assert_eq!(
            unknown.unknown_paths,
            ["crates/new-component/tests/looks_like_a_test.rs"]
        );
        let mixed = analyze_paths([
            "crates/ferrum-cli/tests/a_new_boundary.rs",
            "crates/ferrum-cli/src/commands/run.rs",
        ]);
        assert_eq!(mixed.areas, ALL_AREAS);
        for malformed in [
            "crates/ferrum-cli/tests/.rs",
            "crates/ferrum-cli/tests/readme.md",
        ] {
            assert_eq!(analyze_paths([malformed]).areas, ALL_AREAS);
        }
    }
    #[test]
    fn source_transfer_and_protocol_parsers_keep_their_actual_shared_responsibilities() {
        for path in [
            "crates/ferrum-models/src/hf_download.rs",
            "crates/ferrum-models/src/hf_download/selection.rs",
        ] {
            assert_eq!(analyze_paths([path]).areas, [ChangeArea::Download]);
        }
        let resolver = analyze_paths(["crates/ferrum-cli/src/source_resolver.rs"]);
        assert_eq!(
            resolver.areas,
            [
                ChangeArea::Download,
                ChangeArea::Template,
                ChangeArea::Scheduler,
                ChangeArea::Kv,
            ]
        );
        assert!(!resolver.areas.contains(&ChangeArea::Kernel));
        let adapter = analyze_paths(["crates/ferrum-server/src/axum_server.rs"]);
        assert_eq!(
            adapter.areas,
            [
                ChangeArea::Template,
                ChangeArea::Termination,
                ChangeArea::Structured,
                ChangeArea::Tools,
                ChangeArea::Scheduler
            ]
        );
        for path in [
            "crates/ferrum-engine/src/engine.rs",
            "crates/ferrum-kv/src/managers/paged.rs",
            "crates/ferrum-cli/src/unreviewed.rs",
            "crates/ferrum-server/src/unreviewed.rs",
        ] {
            assert!(
                analyze_paths([path]).areas.contains(&ChangeArea::Kv),
                "{path}"
            );
        }
        for path in [
            "crates/ferrum-types/src/reasoning.rs",
            "crates/ferrum-types/src/reasoning/gemma.rs",
            "crates/ferrum-types/src/harmony.rs",
        ] {
            assert_eq!(
                analyze_paths([path]).areas,
                [
                    ChangeArea::Template,
                    ChangeArea::Termination,
                    ChangeArea::Structured,
                    ChangeArea::Tools
                ]
            );
        }
        let mixed = analyze_paths([
            "crates/ferrum-models/src/hf_download.rs",
            "crates/ferrum-kernels/src/backend/metal/mod.rs",
        ]);
        assert!(mixed.areas.contains(&ChangeArea::Download));
        assert!(mixed.areas.contains(&ChangeArea::Kernel));
        assert!(!mixed.areas.contains(&ChangeArea::Architecture));
        assert_ne!(
            analyze_paths(["crates/ferrum-models/src/hf_download/unknown.rs"]).areas,
            [ChangeArea::Download]
        );
    }
}
