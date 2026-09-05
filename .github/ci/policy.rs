//! Small CI policy tool. Built directly with rustc so planning and the final
//! required check do not compile Ferrum or download Cargo dependencies.
use std::{
    env,
    io::{self, Read},
    process::ExitCode,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scope {
    Docs,
    Code,
}

impl Scope {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "docs" => Ok(Self::Docs),
            "code" => Ok(Self::Code),
            _ => Err(format!("unknown change scope: {value:?}")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Docs => "docs",
            Self::Code => "code",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Success,
    Skipped,
    Failure,
    Cancelled,
}

impl Outcome {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "success" => Ok(Self::Success),
            "skipped" => Ok(Self::Skipped),
            "failure" => Ok(Self::Failure),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(format!("unknown job result: {value:?}")),
        }
    }
}

fn is_documentation(path: &str) -> bool {
    if matches!(path, "README.md" | "README_zh.md" | "CHANGELOG.md") {
        return true;
    }
    let Some(relative) = path.strip_prefix("docs/") else {
        return false;
    };
    relative.ends_with(".md")
        && relative
            .split('/')
            .all(|part| !part.is_empty() && !part.starts_with('.'))
}

fn classify(input: &[u8]) -> Scope {
    // git -z output must retain the final separator. Empty/truncated/unknown
    // input never authorizes skipping the code checks.
    let Some(paths) = input.strip_suffix(&[0]) else {
        return Scope::Code;
    };
    let Ok(paths) = std::str::from_utf8(paths) else {
        return Scope::Code;
    };
    if paths
        .split('\0')
        .all(|path| !path.is_empty() && is_documentation(path))
    {
        Scope::Docs
    } else {
        Scope::Code
    }
}

fn aggregate(prepare: &str, scope: &str, jobs: [&str; 3]) -> Result<(), String> {
    if Outcome::parse(prepare)? != Outcome::Success {
        return Err(format!("prepare must succeed, got {prepare:?}"));
    }
    let expected = match Scope::parse(scope)? {
        Scope::Docs => Outcome::Skipped,
        Scope::Code => Outcome::Success,
    };
    for (name, value) in ["CPU", "Metal", "CUDA"].into_iter().zip(jobs) {
        let actual = Outcome::parse(value).map_err(|error| format!("{name}: {error}"))?;
        if actual != expected {
            return Err(format!(
                "{scope} changes require {name}={expected:?}, got {actual:?}"
            ));
        }
    }
    Ok(())
}

fn run(args: &[String]) -> Result<(), String> {
    match args {
        [command] if command == "classify" => {
            let mut input = Vec::new();
            io::stdin().read_to_end(&mut input).map_err(|e| format!("reading changed paths: {e}"))?;
            println!("{}", classify(&input).as_str());
            Ok(())
        }
        [command, prepare, scope, cpu, metal, cuda] if command == "aggregate" => {
            eprintln!("prepare={prepare:?}, scope={scope:?}, CPU={cpu:?}, Metal={metal:?}, CUDA={cuda:?}");
            aggregate(prepare, scope, [cpu, metal, cuda])
        }
        _ => Err("usage: policy classify < changed-paths.z; policy aggregate PREPARE SCOPE CPU METAL CUDA".to_owned()),
    }
}

fn main() -> ExitCode {
    match run(&env::args().skip(1).collect::<Vec<_>>()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("CI policy: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_docs_can_skip_code_checks() {
        assert_eq!(classify(b"README.md\0README_zh.md\0CHANGELOG.md\0docs/guide.md\0docs/nested/guide with spaces.md\0"), Scope::Docs);
    }

    #[test]
    fn code_fixtures_configuration_and_unknown_paths_require_code_checks() {
        for path in [
            "crates/ferrum-engine/src/lib.rs",
            "crates/ferrum-server/tests/fixture.md",
            "Cargo.lock",
            ".github/workflows/ci.yml",
            "AGENTS.md",
            "docs/sample.json",
            "README.MD",
            "docs/.hidden.md",
            "docs/../crates/fixture.md",
            "docs//guide.md",
            "/docs/guide.md",
        ] {
            let mut paths = b"README.md\0".to_vec();
            paths.extend_from_slice(path.as_bytes());
            paths.push(0);
            assert_eq!(classify(&paths), Scope::Code, "{path}");
        }
    }

    #[test]
    fn incomplete_or_unknown_path_input_does_not_skip_checks() {
        for paths in [
            &b""[..],
            b"\0",
            b"README.md",
            b"README.md\0\0",
            b"README.md\0docs/guide.md",
            b"docs/\xff.md\0",
        ] {
            assert_eq!(classify(paths), Scope::Code, "{paths:?}");
        }
    }

    #[test]
    fn code_renamed_into_docs_still_requires_code_checks() {
        // --no-renames emits both the removed source and added destination.
        assert_eq!(
            classify(b"crates/fixture.md\0docs/fixture.md\0"),
            Scope::Code
        );
        assert_eq!(classify(b"docs/old.md\0docs/new.md\0"), Scope::Docs);
    }

    #[test]
    fn accepts_successful_code_and_expected_documentation_skips() {
        assert!(aggregate("success", "code", ["success"; 3]).is_ok());
        assert!(aggregate("success", "docs", ["skipped"; 3]).is_ok());
    }

    #[test]
    fn every_required_job_must_have_its_expected_result() {
        for (scope, expected) in [("code", "success"), ("docs", "skipped")] {
            for index in 0..3 {
                for wrong in [
                    "success",
                    "skipped",
                    "failure",
                    "cancelled",
                    "neutral",
                    "",
                    "unknown",
                ] {
                    if wrong == expected {
                        continue;
                    }
                    let mut jobs = [expected; 3];
                    jobs[index] = wrong;
                    assert!(
                        aggregate("success", scope, jobs).is_err(),
                        "{scope}: {jobs:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn failed_prepare_and_unknown_scope_cannot_be_hidden_by_skipped_jobs() {
        for prepare in ["failure", "cancelled", "skipped", "neutral", "", "unknown"] {
            assert!(aggregate(prepare, "docs", ["skipped"; 3]).is_err());
            assert!(aggregate(prepare, "code", ["success"; 3]).is_err());
        }
        for scope in ["", "unknown", "docs\n"] {
            assert!(aggregate("success", scope, ["success"; 3]).is_err());
            assert!(aggregate("success", scope, ["skipped"; 3]).is_err());
        }
    }
}
