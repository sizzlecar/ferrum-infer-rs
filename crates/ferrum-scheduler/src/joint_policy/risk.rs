use ferrum_interfaces::request_observation::StoppingForecast;
use ferrum_types::SchedulerSloConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointFeasibility {
    Feasible,
    Infeasible,
    Unresolved,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    pub lower: f64,
    pub upper: f64,
}

/// Conditional outcome at one actual discrete stopping position. ITL here is
/// explicitly an output-commit event proxy, not client-visible SSE evidence.
#[derive(Debug, Clone, Copy)]
pub struct EndpointRisk {
    pub generated_tokens: usize,
    pub ttft_violation: bool,
    pub tpot_violation: Option<bool>,
    pub itl_violations: usize,
    pub itl_events: usize,
    pub completed_by_deadline: bool,
}

/// Endpoints must already reflect a declared joint cost/stopping scenario.
/// Merely multiplying marginal stopping and marginal cost quantiles is invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostStoppingDependence {
    ConditionalEndpointScenario,
    /// Explicit comonotone cost/stopping-rank sensitivity assumption; this
    /// finite empirical approximation is neither independence nor confidence.
    SharedCostStoppingRank,
    Unspecified,
}

pub struct RequestRisk<'a> {
    pub forecast: &'a StoppingForecast,
    pub endpoints: &'a [EndpointRisk],
    pub output_cap: usize,
    pub dependence: CostStoppingDependence,
    /// Separate product success support. Stopping forecasts condition on
    /// successful execution and cannot supply this probability. Absence keeps
    /// completion unresolved even for a deterministic output cap.
    pub product_success: Option<ProbabilityInterval>,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueRisk {
    pub ttft: ProbabilityInterval,
    pub tpot: Option<ProbabilityInterval>,
    pub itl: Option<ProbabilityInterval>,
    pub completion_failure: ProbabilityInterval,
    pub tpot_undefined_mass: f64,
    pub request_count: usize,
}

impl QueueRisk {
    pub fn classify(&self, config: &SchedulerSloConfig) -> JointFeasibility {
        let mut unresolved = false;
        let limits = [
            (Some(self.ttft), 1.0 - config.joint.ttft_quantile),
            (self.tpot, 1.0 - config.joint.tpot_quantile),
            (self.itl, 1.0 - config.joint.itl_quantile),
            (
                Some(self.completion_failure),
                1.0 - config.joint.min_completion_rate,
            ),
        ];
        for (interval, limit) in limits {
            let Some(interval) = interval else {
                continue;
            };
            if interval.lower > limit {
                return JointFeasibility::Infeasible;
            }
            if interval.upper > limit {
                unresolved = true;
            }
        }
        if unresolved {
            JointFeasibility::Unresolved
        } else {
            JointFeasibility::Feasible
        }
    }
}

/// Equal request weights for TTFT/completion and eligible-request weights for
/// TPOT. ITL uses event counts from the same request cohort. Unknown endpoints
/// contribute [0,U], including horizon survival; they are never filled at cap.
pub fn queue_risk(requests: &[RequestRisk<'_>]) -> Option<QueueRisk> {
    if requests.is_empty() {
        return None;
    }
    let (mut ttft, mut completion_lower, mut completion_upper, mut unknown) = (0.0, 0.0, 0.0, 0.0);
    let (mut tpot, mut eligible, mut undefined, mut unknown_eligible) = (0.0, 0.0, 0.0, 0.0);
    let (mut itl, mut events, mut unknown_events) = (0.0, 0.0, 0.0);
    for request in requests {
        let forecast = request.forecast;
        if request.output_cap == 0
            || request.dependence == CostStoppingDependence::Unspecified
            || request.product_success.is_some_and(|success| {
                !success.lower.is_finite()
                    || !success.upper.is_finite()
                    || success.lower < 0.0
                    || success.upper > 1.0
                    || success.lower > success.upper
            })
            || ![
                forecast.known_survival_mass,
                forecast.unresolved_mass,
                forecast.tpot_undefined_mass,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0)
            || forecast.stops.iter().enumerate().any(|(i, stop)| {
                !stop.mass.is_finite()
                    || stop.mass < 0.0
                    || stop.generated_tokens == 0
                    || stop.generated_tokens > request.output_cap
                    || forecast.stops[..i]
                        .iter()
                        .any(|prior| prior.generated_tokens == stop.generated_tokens)
            })
            || (forecast.resolved_mass() + forecast.known_survival_mass + forecast.unresolved_mass
                - 1.0)
                .abs()
                > 1e-9
            || (forecast
                .stops
                .iter()
                .filter(|stop| stop.generated_tokens < 2)
                .map(|stop| stop.mass)
                .sum::<f64>()
                - forecast.tpot_undefined_mass)
                .abs()
                > 1e-9
            || request.endpoints.iter().enumerate().any(|(i, endpoint)| {
                endpoint.itl_violations > endpoint.itl_events
                    || endpoint.itl_events > endpoint.generated_tokens.saturating_sub(1)
                    || endpoint.generated_tokens == 0
                    || endpoint.generated_tokens > request.output_cap
                    || endpoint.tpot_violation.is_some() != (endpoint.generated_tokens >= 2)
                    || request.endpoints[..i]
                        .iter()
                        .any(|prior| prior.generated_tokens == endpoint.generated_tokens)
            })
        {
            return None;
        }
        let mut request_unknown = forecast.known_survival_mass + forecast.unresolved_mass;
        let mut request_deadline_failure = 0.0;
        let mut request_unknown_events =
            request_unknown * request.output_cap.saturating_sub(1) as f64;
        let mut request_unknown_eligible = if request.output_cap >= 2 {
            request_unknown
        } else {
            0.0
        };
        for stop in &forecast.stops {
            if stop.generated_tokens < 2 {
                undefined += stop.mass;
            }
            let Some(endpoint) = request
                .endpoints
                .iter()
                .find(|endpoint| endpoint.generated_tokens == stop.generated_tokens)
            else {
                request_unknown += stop.mass;
                request_unknown_events +=
                    stop.mass * stop.generated_tokens.saturating_sub(1) as f64;
                if stop.generated_tokens >= 2 {
                    request_unknown_eligible += stop.mass;
                }
                continue;
            };
            ttft += stop.mass * f64::from(endpoint.ttft_violation);
            request_deadline_failure += stop.mass * f64::from(!endpoint.completed_by_deadline);
            if let Some(violation) = endpoint.tpot_violation {
                eligible += stop.mass;
                tpot += stop.mass * f64::from(violation);
            }
            itl += stop.mass * endpoint.itl_violations as f64;
            events += stop.mass * endpoint.itl_events as f64;
        }
        unknown += request_unknown;
        unknown_eligible += request_unknown_eligible;
        unknown_events += request_unknown_events;
        let success = request.product_success.unwrap_or(ProbabilityInterval {
            lower: 0.0,
            upper: 1.0,
        });
        // Timing probabilities are conditional on successful execution, so
        // this is the conditional-probability identity, not independence.
        completion_lower += 1.0 - success.upper * (1.0 - request_deadline_failure);
        completion_upper +=
            1.0 - success.lower * (1.0 - request_deadline_failure - request_unknown).max(0.0);
    }
    let count = requests.len() as f64;
    let ratio = |bad: f64, known_denominator: f64, unknown_denominator: f64| {
        let denominator = known_denominator + unknown_denominator;
        (denominator > 0.0).then(|| ProbabilityInterval {
            lower: bad / denominator,
            upper: (bad + unknown_denominator) / denominator,
        })
    };
    Some(QueueRisk {
        ttft: ProbabilityInterval {
            lower: ttft / count,
            upper: (ttft + unknown) / count,
        },
        tpot: ratio(tpot, eligible, unknown_eligible),
        itl: ratio(itl, events, unknown_events),
        completion_failure: ProbabilityInterval {
            lower: completion_lower / count,
            upper: completion_upper / count,
        },
        tpot_undefined_mass: undefined,
        request_count: requests.len(),
    })
}
