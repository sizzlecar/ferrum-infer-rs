//! Conservative component mapping for a complete PR or release diff.
//! This does not infer CUDA/Metal or model-specific reach from a filename. Such
//! changes retain component-wide obligations until a finer mapping is declared.
use super::types::{ChangeArea, Impact, PathImpact};
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
    if path.starts_with(".github/workflows/") || path.starts_with(".github/ci/") {
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
    if matches!(relative, "Cargo.toml" | "build.rs") {
        // A crate dependency or build hook can change behavior beyond its own source.
        return Some((
            ALL_AREAS.to_vec(),
            "crate build/dependency change: conservatively include all components",
        ));
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
            "backend numerical and memory paths; CUDA/Metal scope conservatively not narrowed",
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
        "ferrum-bench-core" | "ferrum-testkit" => (
            vec![Validation],
            "validation oracles, workload selection and measurement infrastructure",
        ),
        _ => return None,
    };
    Some((areas, reason))
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
            "crates/ferrum-bench-core/src/release_regression/selection.rs",
        ]);
        assert!(validation.areas.contains(&ChangeArea::Validation));
        assert!(validation.areas.contains(&ChangeArea::Build));
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
}
