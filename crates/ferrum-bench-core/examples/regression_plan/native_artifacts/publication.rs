//! Prove the finite publication condition without weakening mandatory producers.
//! This accepts only positive required-job success predicates. Monotonicity then
//! makes all-success plus every single-failure state a complete job-state proof;
//! active/cloud/cancellation states are enumerated independently.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conclusion {
    Success,
    Skipped,
    Other,
}

#[derive(Debug)]
enum Predicate {
    Always,
    NotCancelled,
    Active(bool),
    Cloud(bool),
    Result(usize, Conclusion),
    And(Box<Self>, Box<Self>),
    Or(Box<Self>, Box<Self>),
}

struct State {
    active: Option<bool>,
    cloud: Option<bool>,
    cloud_job: usize,
    cloud_result: Conclusion,
    cancelled: bool,
    failed_job: Option<usize>,
}

impl Predicate {
    fn evaluate(&self, state: &State) -> bool {
        match self {
            Self::Always => true,
            Self::NotCancelled => !state.cancelled,
            Self::Active(expected) => state.active == Some(*expected),
            Self::Cloud(expected) => state.cloud == Some(*expected),
            Self::Result(job, expected) if *job == state.cloud_job => {
                state.cloud_result == *expected
            }
            Self::Result(job, _) => state.failed_job != Some(*job),
            Self::And(left, right) => left.evaluate(state) && right.evaluate(state),
            Self::Or(left, right) => left.evaluate(state) || right.evaluate(state),
        }
    }
}

struct Parser<'a> {
    remaining: &'a str,
    jobs: &'a [String],
    cloud_job: usize,
}

impl Parser<'_> {
    fn take(&mut self, token: &str) -> bool {
        self.remaining = self.remaining.trim_start();
        if let Some(rest) = self.remaining.strip_prefix(token) {
            self.remaining = rest;
            true
        } else {
            false
        }
    }

    fn require(&mut self, token: &str) -> Result<(), String> {
        if self.take(token) {
            Ok(())
        } else {
            Err(format!(
                "unsupported publication predicate: expected {token}"
            ))
        }
    }

    fn expression(&mut self) -> Result<Predicate, String> {
        let mut value = self.conjunction()?;
        while self.take("||") {
            value = Predicate::Or(Box::new(value), Box::new(self.conjunction()?));
        }
        Ok(value)
    }

    fn conjunction(&mut self) -> Result<Predicate, String> {
        let mut value = self.atom()?;
        while self.take("&&") {
            value = Predicate::And(Box::new(value), Box::new(self.atom()?));
        }
        Ok(value)
    }

    fn atom(&mut self) -> Result<Predicate, String> {
        if self.take("(") {
            let value = self.expression()?;
            self.require(")")?;
            return Ok(value);
        }
        if self.take("!") {
            self.require("cancelled()")?;
            return Ok(Predicate::NotCancelled);
        }
        if self.take("always()") {
            return Ok(Predicate::Always);
        }
        self.remaining = self.remaining.trim_start();
        let end = self
            .remaining
            .find(|ch: char| !ch.is_ascii_alphanumeric() && !matches!(ch, '.' | '_' | '-'))
            .unwrap_or(self.remaining.len());
        let name = &self.remaining[..end];
        self.remaining = &self.remaining[end..];
        self.require("==")?;
        self.require("'")?;
        let end = self
            .remaining
            .find('\'')
            .ok_or("unclosed predicate value")?;
        let expected = &self.remaining[..end];
        self.remaining = &self.remaining[end + 1..];
        let boolean = || match expected {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err("unsupported publication output value".to_string()),
        };
        match name {
            "needs.prepare.outputs.active" => return Ok(Predicate::Active(boolean()?)),
            "needs.prepare.outputs.cloud_cuda" => return Ok(Predicate::Cloud(boolean()?)),
            _ => {}
        }
        let name = name
            .strip_prefix("needs.")
            .and_then(|name| name.strip_suffix(".result"))
            .ok_or("unsupported publication operand")?;
        let job = self
            .jobs
            .iter()
            .position(|job| job == name)
            .ok_or("publication predicate references an undeclared dependency")?;
        let result = match expected {
            "success" => Conclusion::Success,
            "skipped" if job == self.cloud_job => Conclusion::Skipped,
            _ => return Err("required publication jobs must positively require success".into()),
        };
        Ok(Predicate::Result(job, result))
    }
}

fn parse(job: &Value) -> Result<(Predicate, Vec<String>, usize), String> {
    let jobs: Vec<String> = serde_json::from_value(job["needs"].clone())
        .map_err(|_| "publication needs must be explicit job names")?;
    if jobs.iter().collect::<BTreeSet<_>>().len() != jobs.len()
        || !jobs.iter().any(|job| job == "prepare")
        || !jobs.iter().any(|job| job == "cuda-models")
    {
        return Err("publication must uniquely require preparation and local CUDA".into());
    }
    let cloud_job = jobs
        .iter()
        .position(|job| job == "cuda-models-cloud")
        .ok_or("conditional publication has no explicit cloud dependency")?;
    let mut parser = Parser {
        remaining: job["if"]
            .as_str()
            .ok_or("publication condition must be an expression")?,
        jobs: &jobs,
        cloud_job,
    };
    let expression = parser.expression()?;
    if !parser.remaining.trim().is_empty() {
        return Err("unsupported publication expression suffix".into());
    }
    Ok((expression, jobs, cloud_job))
}

pub(super) fn verify(job: &Value) -> Result<(), String> {
    let (expression, jobs, cloud_job) = parse(job)?;
    for active in [None, Some(false), Some(true)] {
        for cloud in [None, Some(false), Some(true)] {
            for cancelled in [false, true] {
                for cloud_result in [Conclusion::Success, Conclusion::Skipped, Conclusion::Other] {
                    for failed_job in std::iter::once(None)
                        .chain((0..jobs.len()).filter(|job| *job != cloud_job).map(Some))
                    {
                        let state = State {
                            active,
                            cloud,
                            cloud_job,
                            cloud_result,
                            cancelled,
                            failed_job,
                        };
                        let expected = active == Some(true)
                            && !cancelled
                            && failed_job.is_none()
                            && matches!(
                                (cloud, cloud_result),
                                (Some(false), Conclusion::Skipped)
                                    | (Some(true), Conclusion::Success)
                            );
                        if expression.evaluate(&state) != expected {
                            return Err("publication condition does not preserve mandatory success and explicit cloud mode".into());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual() -> Value {
        yaml(include_str!(
            "../../../../../.github/workflows/release-delivery.yml"
        ))
        .unwrap()["jobs"]["publish"]
            .clone()
    }

    #[test]
    fn actual_publication_accepts_only_complete_local_and_requested_cloud_evidence() {
        let job = actual();
        verify(&job).unwrap();
        let (expression, jobs, cloud_job) = parse(&job).unwrap();
        // Failure, cancellation, timeout and skip are equivalent for every
        // required job because the parser accepts only its positive success test.
        for failed_job in 0..jobs.len() {
            if failed_job == cloud_job {
                continue;
            }
            for cloud in [false, true] {
                assert!(
                    !expression.evaluate(&State {
                        active: Some(true),
                        cloud: Some(cloud),
                        cloud_job,
                        cloud_result: if cloud {
                            Conclusion::Success
                        } else {
                            Conclusion::Skipped
                        },
                        cancelled: false,
                        failed_job: Some(failed_job),
                    }),
                    "required dependency {} was bypassed",
                    jobs[failed_job]
                );
            }
        }
        let mut reordered = job.clone();
        let source = job["if"].as_str().unwrap();
        reordered["if"] = json!(format!("({source}) && always()"));
        verify(&reordered).unwrap();
    }

    #[test]
    fn weakened_or_unknown_publication_conditions_never_narrow_source_scope() {
        let job = actual();
        let source = job["if"].as_str().unwrap();
        let (_, jobs, cloud_job) = parse(&job).unwrap();
        for (index, name) in jobs.iter().enumerate() {
            if index == cloud_job {
                continue;
            }
            let mut missing = job.clone();
            missing["if"] =
                json!(source.replace(&format!("needs.{name}.result == 'success'"), "always()"));
            assert!(verify(&missing).is_err(), "omitted guard for {name}");
        }
        for condition in [
            format!("({source}) || always()"),
            format!("({source}) && needs.unreviewed.result == 'success'"),
            source.replace("!cancelled()", "always()"),
            source.replace("needs.prepare.outputs.active == 'true'", "always()"),
            source.replace("needs.cuda-models-cloud.result == 'skipped'", "always()"),
            source.replace("needs.cuda-models-cloud.result == 'success'", "always()"),
        ] {
            let mut changed = job.clone();
            changed["if"] = json!(condition);
            assert!(verify(&changed).is_err());
        }
    }
}
