use super::*;

pub(super) struct Control {
    margin: u64,
    residual_ns: u64,
    span: u64,
    floor: u64,
    max_wave: u64,
    residual_query: Box<dyn Fn(&StructuredQueryV2) -> bool>,
    qualification_query: Option<Box<dyn Fn(&StructuredQueryV2) -> bool>>,
    q_time: Option<u64>,
    q: Errors,
    q_by_rows: BTreeMap<u32, Errors>,
    q_gate_unknown: usize,
    q_misses: Vec<Value>,
    future: BTreeMap<u32, Future>,
    paired: Vec<Value>,
}

impl Control {
    pub(super) fn calibrate(
        fit: &FittedStructuredModelV2,
        samples: &[StructuredNumericObservationV2],
        settings: &StructuredSettingsV2,
    ) -> Result<Self, String> {
        let mut errors = Vec::with_capacity(samples.len());
        for sample in samples {
            let point = fit
                .diagnose_original_actual_point(&sample.input)
                .ok_or("original R input point unavailable")?;
            errors.push(i128::from(sample.wall_ns) - i128::from(point));
        }
        errors.sort_unstable();
        let residual_ns =
            u64::try_from(errors[(99 * errors.len()).div_ceil(100) - 1].max(0)).unwrap();
        let span = match settings.learned_drift {
            StructuredLearnedDriftV2::Disabled => 0,
            StructuredLearnedDriftV2::ObservedResidualSpanV1 {
                maximum_span_margin_ns,
            } => {
                let span = u64::try_from(errors.last().unwrap() - errors[0]).unwrap();
                if span > maximum_span_margin_ns.get() {
                    return Err("original learned span limit".into());
                }
                span
            }
        };
        let (floor, _) = fit.diagnose_empirical_fit_limits();
        let margin = residual_ns
            .max(floor)
            .checked_add(span)
            .and_then(|v| v.checked_add(settings.static_margin_ns))
            .ok_or("margin overflow")?;
        Ok(Self {
            margin,
            residual_ns,
            span,
            floor,
            max_wave: settings.max_wave_ns,
            residual_query: fit
                .diagnose_frozen_phase_query_gate(samples)
                .ok_or("R branch coverage")?,
            qualification_query: None,
            q_time: None,
            q: Errors::default(),
            q_by_rows: BTreeMap::new(),
            q_gate_unknown: 0,
            q_misses: Vec::new(),
            future: BTreeMap::new(),
            paired: Vec::new(),
        })
    }

    pub(super) fn qualify(
        &mut self,
        fit: &FittedStructuredModelV2,
        samples: &[StructuredNumericObservationV2],
        at_ns: u64,
    ) {
        self.q_time = Some(at_ns);
        for sample in samples {
            let query = StructuredQueryV2::exact(sample.input.clone());
            let point = fit.diagnose_fit_covered_point(&query, at_ns);
            let prediction = point
                .and_then(|p| p.checked_add(self.margin).map(|v| (p, v)))
                .filter(|(_, v)| *v <= self.max_wave);
            if !(self.residual_query)(&query) || prediction.is_none() {
                self.q_gate_unknown += 1;
                self.q_misses.push(
                    json!({"call":sample.call_id,"rows":sample.input.owner().rows,
                    "failure":"original Fit/R/clock/numeric-cap gate"}),
                );
                continue;
            }
            let (point, planning) = prediction.unwrap();
            self.q.observe(point, planning, sample.wall_ns);
            self.q_by_rows
                .entry(sample.input.owner().rows)
                .or_default()
                .observe(point, planning, sample.wall_ns);
            if planning < sample.wall_ns {
                self.q_misses.push(
                    json!({"call":sample.call_id,"rows":sample.input.owner().rows,
                    "wall_ns":sample.wall_ns,"point_ns":point,"planning_ns":planning,
                    "under_ns":sample.wall_ns-planning}),
                );
            }
        }
        self.qualification_query = fit.diagnose_frozen_phase_query_gate(samples);
        assert_eq!(
            self.q.count + self.q_gate_unknown,
            samples.len(),
            "every original selected Q member accounted"
        );
    }

    fn q_passed(&self) -> bool {
        self.q_time.is_some()
            && self.q_gate_unknown == 0
            && self.q.misses == 0
            && self.qualification_query.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn future(
        &mut self,
        fit: &FittedStructuredModelV2,
        query: &StructuredQueryV2,
        now: u64,
        call: u64,
        wall: u64,
        paired: bool,
    ) {
        let rows = query.input().owner().rows;
        let candidate_passed = self.q_passed();
        let query_time = self.q_time.is_some_and(|q| now >= q);
        let point = fit.diagnose_fit_covered_point(query, now);
        let residual_ok = (self.residual_query)(query);
        let qualification_ok = self.qualification_query.as_ref().is_some_and(|g| g(query));
        let prediction = point
            .and_then(|p| p.checked_add(self.margin).map(|v| (p, v)))
            .filter(|(_, v)| *v <= self.max_wave);
        let count = self.future.entry(rows).or_default();
        count.total += 1;
        if query_time && residual_ok && qualification_ok && prediction.is_some() {
            let (point, planning) = prediction.unwrap();
            count.original_structural_gate_accepted += 1;
            count.errors.observe(point, planning, wall);
            if paired {
                self.paired
                    .push(json!({"call":call,"rows":rows,"demand_verified":true,
                "original_structural_gates":true,"counterfactual_full_q_pass":candidate_passed,
                "point_ns":point,"margin_ns":self.margin,"planning_ns":planning,"wall_ns":wall,
                "underestimate":planning<wall}));
            }
        } else {
            let reason=format!("time={query_time};fit_point={};R={residual_ok};Q={qualification_ok};numeric_cap={}",
                point.is_some(),prediction.is_some());
            *count.unknown.entry(reason.clone()).or_default() += 1;
            if paired {
                self.paired
                    .push(json!({"call":call,"rows":rows,"demand_verified":true,
                "original_structural_gates":false,"reason":reason}));
            }
        }
    }
    pub(super) fn report(&self) -> Value {
        json!({"margin_ns":self.margin,"residual_p99_ns":self.residual_ns,"learned_span_ns":self.span,
            "original_fit_floor_ns":self.floor,"full_q_pass":self.q_passed(),
            "q_time_ns":self.q_time,"q_errors":self.q.summary(),"q_gate_unknown":self.q_gate_unknown,
            "q_by_rows":self.q_by_rows.iter().map(|(r,v)|json!({"rows":r,"errors":v.summary()})).collect::<Vec<_>>(),
            "all_q_failures":self.q_misses,
            "future_by_rows":self.future.iter().map(|(r,v)|json!({"rows":r,"total":v.total,
                "original_structural_gate_accepted":v.original_structural_gate_accepted,
                "unknown":v.unknown,"errors":v.errors.summary()})).collect::<Vec<_>>(),
            "paired_actual_demands":self.paired,
            "authority":"counterfactual empirical numbers only; original ChallengeCoverage is axis/branch presence, not joint numeric-range support; no live publication or mathematical bound"})
    }
}
#[derive(Default)]
struct Future {
    total: usize,
    original_structural_gate_accepted: usize,
    unknown: BTreeMap<String, usize>,
    errors: Errors,
}
#[derive(Default)]
struct Errors {
    count: usize,
    misses: usize,
    absolute: Vec<u64>,
    point_absolute: Vec<u64>,
    under: Vec<u64>,
    plans: Vec<u64>,
}
impl Errors {
    fn observe(&mut self, point: u64, plan: u64, wall: u64) {
        self.count += 1;
        self.misses += usize::from(plan < wall);
        self.absolute.push(plan.abs_diff(wall));
        self.point_absolute.push(point.abs_diff(wall));
        self.under.push(wall.saturating_sub(plan));
        self.plans.push(plan);
    }
    fn summary(&self) -> Value {
        fn q(v: &[u64]) -> Option<[u64; 3]> {
            if v.is_empty() {
                return None;
            }
            let mut a = v.to_vec();
            a.sort_unstable();
            Some([
                a[(50 * a.len()).div_ceil(100) - 1],
                a[(99 * a.len()).div_ceil(100) - 1],
                *a.last().unwrap(),
            ])
        }
        json!({"count":self.count,"misses":self.misses,"planning_p50_p99_max_ns":q(&self.plans),
            "point_abs_error_p50_p99_max_ns":q(&self.point_absolute),
            "planning_abs_error_p50_p99_max_ns":q(&self.absolute),"under_p50_p99_max_ns":q(&self.under)})
    }
}
