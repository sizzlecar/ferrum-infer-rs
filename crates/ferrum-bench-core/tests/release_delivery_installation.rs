//! Exercise the committed installation guards over job outcomes, including the
//! implicit Actions success() that otherwise suppresses skipped-ancestor jobs.
use serde_yaml::Value;
use std::collections::{BTreeMap, BTreeSet};
use syn::{BinOp, Expr, Lit, Member, UnOp};

const INSTALLERS: &[&str] = &[
    "install-cargo",
    "install-homebrew",
    "install-bootstrap-unix",
    "install-bootstrap-windows",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResultState {
    Success,
    Failure,
    Cancelled,
    Skipped,
}

impl ResultState {
    const ALL: [Self; 4] = [Self::Success, Self::Failure, Self::Cancelled, Self::Skipped];

    fn text(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Cancelled => "cancelled",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Clone, Debug)]
struct State {
    cancelled: bool,
    ancestors_succeeded: bool,
    active: &'static str,
    verify_only: &'static str,
    results: BTreeMap<String, ResultState>,
}

impl State {
    fn new(active: &'static str, verify_only: &'static str) -> Self {
        Self {
            cancelled: false,
            ancestors_succeeded: false, // optional cloud ancestor was skipped
            active,
            verify_only,
            results: BTreeMap::from([
                ("prepare".into(), ResultState::Success),
                (
                    "publish".into(),
                    if verify_only == "true" {
                        ResultState::Skipped
                    } else {
                        ResultState::Success
                    },
                ),
            ]),
        }
    }

    fn operand(&self, name: &str) -> Result<String, String> {
        let value = match name {
            "needs.prepare.outputs.active" => self.active,
            "needs.prepare.outputs.verify_only" => self.verify_only,
            "needs.prepare.outputs.cloud_cuda" => "false",
            "needs.prepare.outputs.version" => "0.10.0",
            _ => {
                let job = name
                    .strip_prefix("needs.")
                    .and_then(|s| s.strip_suffix(".result"))
                    .ok_or_else(|| format!("unsupported operand {name}"))?;
                self.results
                    .get(&job.replace('_', "-"))
                    .ok_or_else(|| format!("missing job outcome {job}"))?
                    .text()
            }
        };
        Ok(value.into())
    }
}

fn workflow() -> Value {
    serde_yaml::from_str(include_str!(
        "../../../.github/workflows/release-delivery.yml"
    ))
    .unwrap()
}

fn field_path(expr: &Expr) -> Result<String, String> {
    match expr {
        Expr::Path(path) if path.path.segments.len() == 1 => {
            Ok(path.path.segments[0].ident.to_string())
        }
        Expr::Field(field) => match &field.member {
            Member::Named(name) => Ok(format!("{}.{name}", field_path(&field.base)?)),
            _ => Err("indexed workflow operand is not supported".into()),
        },
        _ => Err("unsupported workflow operand".into()),
    }
}

// Reuse the existing Rust expression parser, not a second Actions grammar. Only
// the finite boolean/equality subset used by these guards is accepted below.
fn expression(source: &str) -> Result<Expr, String> {
    let source = source.trim();
    let source = source
        .strip_prefix("${{")
        .and_then(|s| s.strip_suffix("}}"))
        .unwrap_or(source);
    let mut normalized = source.replace('\'', "\"");
    for (job, _) in workflow()["jobs"].as_mapping().unwrap() {
        let job = job.as_str().unwrap();
        normalized = normalized.replace(
            &format!("needs.{job}."),
            &format!("needs.{}.", job.replace('-', "_")),
        );
    }
    syn::parse_str(&normalized).map_err(|error| error.to_string())
}

fn operand(expr: &Expr, state: &State) -> Result<String, String> {
    match expr {
        Expr::Field(_) => state.operand(&field_path(expr)?),
        Expr::Lit(expr) => match &expr.lit {
            Lit::Str(value) => Ok(value.value()),
            _ => Err("equality requires string operands".into()),
        },
        _ => Err("unsupported equality operand".into()),
    }
}

fn evaluate(expr: &Expr, state: &State, explicit_status: &mut bool) -> Result<bool, String> {
    match expr {
        Expr::Paren(expr) => evaluate(&expr.expr, state, explicit_status),
        Expr::Lit(expr) => match &expr.lit {
            Lit::Bool(value) => Ok(value.value),
            _ => Err("unsupported guard literal".into()),
        },
        Expr::Unary(expr) if matches!(expr.op, UnOp::Not(_)) => {
            Ok(!evaluate(&expr.expr, state, explicit_status)?)
        }
        Expr::Binary(expr) if matches!(expr.op, BinOp::Eq(_) | BinOp::Ne(_)) => {
            let equal = operand(&expr.left, state)? == operand(&expr.right, state)?;
            Ok(if matches!(expr.op, BinOp::Eq(_)) {
                equal
            } else {
                !equal
            })
        }
        Expr::Binary(expr) => {
            // Visit both branches even when the left value would short-circuit:
            // unknown operands must fail closed and status detection is syntactic.
            let left = evaluate(&expr.left, state, explicit_status)?;
            let right = evaluate(&expr.right, state, explicit_status)?;
            Ok(match expr.op {
                BinOp::And(_) => left && right,
                BinOp::Or(_) => left || right,
                _ => return Err("unsupported guard operator".into()),
            })
        }
        Expr::Call(expr) if expr.args.is_empty() => {
            let value = match field_path(&expr.func)?.as_str() {
                "always" => true,
                "cancelled" => state.cancelled,
                "success" => state.ancestors_succeeded,
                _ => return Err("unsupported status function".into()),
            };
            *explicit_status = true;
            Ok(value)
        }
        _ => Err("unsupported workflow expression".into()),
    }
}

fn scheduled(expr: &Expr, state: &State) -> Result<bool, String> {
    let mut explicit_status = false;
    let condition = evaluate(expr, state, &mut explicit_status)?;
    Ok(condition && (explicit_status || state.ancestors_succeeded))
}

fn expected_install(state: &State) -> bool {
    !state.cancelled
        && state.results["prepare"] == ResultState::Success
        && matches!(
            (state.active, state.verify_only, state.results["publish"]),
            ("true", "false", ResultState::Success) | ("false", "true", ResultState::Skipped)
        )
}

fn verify_installer(job: &Value) -> Result<(), String> {
    let guard = expression(
        job["if"]
            .as_str()
            .ok_or("missing explicit installation guard")?,
    )?;
    for active in ["", "false", "true"] {
        for verify_only in ["", "false", "true"] {
            for cancelled in [false, true] {
                for ancestors_succeeded in [false, true] {
                    for prepare in ResultState::ALL {
                        for publish in ResultState::ALL {
                            let mut state = State::new(active, verify_only);
                            state.cancelled = cancelled;
                            state.ancestors_succeeded = ancestors_succeeded;
                            state.results.insert("prepare".into(), prepare);
                            state.results.insert("publish".into(), publish);
                            if scheduled(&guard, &state)? != expected_install(&state) {
                                return Err(format!("installation outcome mismatch: {state:?}"));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn public_installation_guards_handle_optional_skip_and_both_delivery_modes() {
    let workflow = workflow();
    for name in INSTALLERS {
        let job = &workflow["jobs"][*name];
        verify_installer(job).unwrap_or_else(|error| panic!("{name}: {error}"));
        let needs = dependencies(job);
        assert!(needs.contains(&"prepare".into()) && needs.contains(&"publish".into()));
        let checkout = job["steps"]
            .as_sequence()
            .unwrap()
            .iter()
            .find(|step| {
                step["uses"]
                    .as_str()
                    .is_some_and(|action| action.starts_with("actions/checkout@"))
            })
            .expect("public installation must check out its verified source");
        assert_eq!(
            checkout["with"]["ref"].as_str(),
            Some("${{ needs.prepare.outputs.source_ref }}")
        );
    }
}

#[test]
fn installation_regressions_cannot_hide_behind_implicit_success_or_weakened_guards() {
    let original = workflow()["jobs"]["install-cargo"].clone();
    let source = original["if"].as_str().unwrap();
    let mut equivalent = original.clone();
    equivalent["if"] = Value::String(format!("({source}) && always()"));
    verify_installer(&equivalent).unwrap();
    for changed in [
        source
            .replace("always()", "true")
            .replace("!cancelled()", "true"),
        source.replace("!cancelled()", "true"),
        source.replace("needs.prepare.result == 'success'", "true"),
        source.replace("needs.publish.result == 'success'", "true"),
        source.replace("needs.publish.result == 'skipped'", "true"),
    ] {
        let mut job = original.clone();
        job["if"] = Value::String(changed.clone());
        assert!(
            verify_installer(&job).is_err(),
            "accepted weakened guard: {changed}"
        );
    }
    let formal = State::new("true", "false");
    assert!(!scheduled(
        &expression("needs.publish.result == 'success'").unwrap(),
        &formal
    )
    .unwrap());
    assert!(scheduled(&expression(source).unwrap(), &formal).unwrap());
    assert!(expression("unreviewed()")
        .and_then(|expr| scheduled(&expr, &formal))
        .is_err());
}

fn dependencies(job: &Value) -> Vec<String> {
    match &job["needs"] {
        Value::String(name) => vec![name.clone()],
        Value::Sequence(names) => names
            .iter()
            .map(|name| name.as_str().unwrap().into())
            .collect(),
        Value::Null => Vec::new(),
        _ => panic!("unsupported job dependency shape"),
    }
}

#[test]
fn verification_only_cannot_schedule_staging_models_cloud_or_publication() {
    let workflow = workflow();
    let jobs = workflow["jobs"].as_mapping().unwrap();
    let mut state = State::new("false", "true");
    state.results.remove("publish");
    let mut pending: BTreeSet<String> = jobs
        .keys()
        .map(|key| key.as_str().unwrap().to_owned())
        .filter(|name| {
            name != "prepare" && name != "complete" && !INSTALLERS.contains(&name.as_str())
        })
        .collect();
    while !pending.is_empty() {
        let ready: Vec<_> = pending
            .iter()
            .filter(|name| {
                dependencies(&jobs[name.as_str()])
                    .iter()
                    .all(|need| state.results.contains_key(need))
            })
            .cloned()
            .collect();
        assert!(
            !ready.is_empty(),
            "cyclic or missing dependencies: {pending:?}"
        );
        for name in ready {
            let job = &jobs[name.as_str()];
            state.ancestors_succeeded = dependencies(job)
                .iter()
                .all(|need| state.results[need] == ResultState::Success);
            let guard = expression(job["if"].as_str().unwrap_or("success()")).unwrap();
            assert!(
                !scheduled(&guard, &state).unwrap(),
                "verification scheduled heavy job {name}"
            );
            state.results.insert(name.clone(), ResultState::Skipped);
            pending.remove(&name);
        }
    }
    assert_eq!(state.results["publish"], ResultState::Skipped);
    for name in INSTALLERS {
        assert!(scheduled(
            &expression(jobs[*name]["if"].as_str().unwrap()).unwrap(),
            &state
        )
        .unwrap());
    }
}

#[cfg(unix)]
fn complete_accepts(workflow: &Value, state: &State) -> bool {
    use std::process::Command;
    let job = &workflow["jobs"]["complete"];
    if !scheduled(&expression(job["if"].as_str().unwrap()).unwrap(), state).unwrap() {
        return false;
    }
    let temporary = tempfile::tempdir().unwrap();
    let step = &job["steps"].as_sequence().unwrap()[0];
    let mut command = Command::new("bash");
    command
        .args([
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            step["run"].as_str().unwrap(),
        ])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GITHUB_STEP_SUMMARY", temporary.path().join("summary"))
        .current_dir(temporary.path());
    for (key, value) in step["env"].as_mapping().unwrap() {
        let field = value
            .as_str()
            .unwrap()
            .trim()
            .strip_prefix("${{")
            .unwrap()
            .strip_suffix("}}")
            .unwrap()
            .trim();
        let value = state.operand(&field.replace('-', "_")).unwrap();
        command.env(key.as_str().unwrap(), value);
    }
    command
        .output()
        .expect("execute only committed completion guard")
        .status
        .success()
}

#[cfg(unix)]
#[test]
fn complete_requires_the_correct_publication_mode_and_every_public_installation() {
    let workflow = workflow();
    for (active, verify_only) in [("true", "false"), ("false", "true")] {
        let mut state = State::new(active, verify_only);
        for installer in INSTALLERS {
            state
                .results
                .insert((*installer).into(), ResultState::Success);
        }
        assert!(complete_accepts(&workflow, &state), "{state:?}");
        for name in std::iter::once("prepare")
            .chain(std::iter::once("publish"))
            .chain(INSTALLERS.iter().copied())
        {
            for result in ResultState::ALL {
                if result == state.results[name] {
                    continue;
                }
                let mut failed = state.clone();
                failed.results.insert(name.into(), result);
                assert!(
                    !complete_accepts(&workflow, &failed),
                    "accepted {name}={result:?}, mode={verify_only}"
                );
            }
        }
        state.cancelled = true;
        assert!(
            !complete_accepts(&workflow, &state),
            "cancelled workflow was accepted"
        );
    }
    for active in ["", "false", "true"] {
        for verify_only in ["", "false", "true"] {
            if matches!((active, verify_only), ("true", "false") | ("false", "true")) {
                continue;
            }
            let mut invalid = State::new(active, verify_only);
            for installer in INSTALLERS {
                invalid
                    .results
                    .insert((*installer).into(), ResultState::Success);
            }
            assert!(
                !complete_accepts(&workflow, &invalid),
                "ambiguous mode: {invalid:?}"
            );
        }
    }
}
