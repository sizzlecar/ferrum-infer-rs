//! Original worker collector. MemoryOnly hashes canonical records without IO.
use super::*;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::StructuredPhaseV2,
    cost_profile::{
        self as file, StructuredServiceClockV6, StructuredServiceCollectorV6,
        StructuredServiceDeclarationV6, StructuredServiceHeaderV6, StructuredServiceRecordV6,
    },
};

#[cfg(test)]
#[path = "session/test_fixtures.rs"]
mod test_fixtures;

pub(super) struct Session {
    collector: StructuredServiceCollectorV6,
    declaration: Declaration,
    generation: u64,
    pub(super) phase: usize,
    source: Option<super::source::Source>,
    shadow: super::shadow::ShadowDiscovery,
    discovery_policy: discovery::DiscoveryPolicy,
}

pub(super) enum PhaseCompletion {
    Continue,
    AllChildrenFailed(Option<super::shadow::ShadowDiscoverySeed>),
}

pub(super) fn phase(index: usize) -> StructuredPhaseV2 {
    match index {
        0 => StructuredPhaseV2::Fit,
        1 => StructuredPhaseV2::Residual,
        _ => StructuredPhaseV2::Qualification,
    }
}
pub(super) fn paired(value: ExportClockReading) -> StructuredServiceClockV6 {
    StructuredServiceClockV6 {
        wall_unix_ns: value.wall_unix_ns,
        monotonic_ns: value.monotonic_ns,
    }
}

impl Session {
    pub(super) fn open(
        live: &LiveCalibration,
        declaration: Declaration,
        generation: u64,
        opening: ExportClockReading,
        cutoff: u64,
        import: &ferrum_types::SloCostProfileImportConfig,
        diagnostics: Option<&diagnostics::Store>,
        discovery_policy: discovery::DiscoveryPolicy,
    ) -> Result<Self, FerrumError> {
        use sha2::Digest;
        let header = StructuredServiceHeaderV6::new(
            sha2::Sha256::digest(uuid::Uuid::new_v4().as_bytes()).into(),
            generation,
            file::ProfileFingerprint::from(&live.fingerprint),
            serde_json::to_value(live.producer.current()?).map_err(error)?,
            paired(opening),
            StructuredServiceDeclarationV6 {
                route_population: live.route_population(),
                domain_policy: if live.workload_domain.is_some() {
                    ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
                } else {
                    ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredServiceDomainPolicyV1::FrozenFitSupportV1
                },
                nonnegative_envelope: live.workload_domain.as_ref().map(|domain| {
                    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
                        NonNegativeEnvelopeContractV1, WorkAxisAndBranchChallengesV1,
                    };
                    NonNegativeEnvelopeContractV1 {
                        algorithm_universe: None,
                        population_policy: Default::default(),
planning_estimator: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1::IdentifiedEnvelopeV2,
                        template_policy: ferrum_scheduler::implementations::continuous::cost_model::structured_v2::StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
                        workload_domain: domain.clone(),
                        settings: Default::default(),
                        challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
                    }
                }),
                phase_offered_waves: declaration.phase_offered_waves,
                maximum_window_ns: declaration.maximum_window_ns,
                settings: declaration.settings.clone(),
                scopes: declaration.scopes.clone(),
                maximum_retained_numeric_bytes: live.maximum_retained_bytes,
            },
            import.max_file_bytes.get() as u64,
        )
        .map_err(error)?;
        let mut source = diagnostics.and_then(|store| {
            match super::source::Source::open(live, generation, store, &header) {
                Ok(source) => Some(source),
                Err(reason) => {
                    live.record_automatic_diagnostic_failure(
                        generation,
                        super::source::DiagnosticFailureStage::Open,
                        &reason,
                    );
                    None
                }
            }
        });
        let collector = StructuredServiceCollectorV6::new(header, profile::load_limits(import))
            .map_err(|reason| {
                if let Some(source) = &source {
                    source.note_failure(live, generation, &reason.to_string());
                }
                error(reason)
            })?;
        let shadow = super::shadow::ShadowDiscovery::open(
            discovery_policy,
            declaration.phase_offered_waves[0],
            cutoff,
        );
        let mut session = Self {
            collector,
            declaration,
            generation,
            phase: 0,
            source: source.take(),
            shadow,
            discovery_policy,
        };
        if let Err(reason) = session.record(
            live,
            &StructuredServiceRecordV6::PhaseOpen {
                phase: phase(0),
                opened_at_ns: opening.monotonic_ns,
                fifo_cutoff: cutoff,
            },
        ) {
            if let Some(source) = &session.source {
                source.note_failure(live, generation, &reason.to_string());
            }
            return Err(reason);
        }
        Ok(session)
    }

    pub(super) fn observe_shadow(
        &mut self,
        ticket: u64,
        fifo: u64,
        input: Option<&StructuredInputV2>,
    ) {
        self.shadow.observe(ticket, fifo, input);
    }

    fn record(
        &mut self,
        live: &LiveCalibration,
        record: &StructuredServiceRecordV6,
    ) -> Result<(), FerrumError> {
        self.collector.push(record).map_err(error)?;
        self.write_diagnostic(live, record);
        Ok(())
    }

    fn write_diagnostic(&mut self, live: &LiveCalibration, record: &impl Serialize) {
        if let Some(source) = &mut self.source {
            if let Err(reason) = source.write(record) {
                source.note_failure(live, self.generation, &reason.to_string());
                live.record_automatic_diagnostic_failure(
                    self.generation,
                    super::source::DiagnosticFailureStage::Write,
                    &reason,
                );
                // Preserve the incomplete original artifact, but never resume
                // writing after a gap or present it as a complete source.
                self.source = None;
            }
        }
    }

    pub(super) fn complete_phase(
        &mut self,
        live: &LiveCalibration,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<PhaseCompletion, FerrumError> {
        if window.generation != self.generation || window.phase != self.phase || self.phase >= 3 {
            return Err(error("numerical generation/phase mismatch"));
        }
        let now = clock
            .now_ns()
            .ok_or_else(|| error("phase clock unavailable"))?;
        let mut collected = live.collected.lock();
        if collected.failure.is_some()
            || collected.waves.len() != self.declaration.phase_offered_waves[self.phase]
        {
            return Err(error("numerical offered population is incomplete"));
        }
        collected.waves.sort_by_key(|wave| wave.ticket);
        let offset = self.declaration.phase_offered_waves[..self.phase]
            .iter()
            .try_fold(0u64, |total, count| total.checked_add(*count as u64))
            .ok_or_else(|| error("offered population overflow"))?;
        for wave in &collected.waves {
            self.record(
                live,
                &publication::completed_record(wave, self.phase, offset)?,
            )?;
            if let Some(source) = &mut self.source {
                source.completed(wave.ticket);
            }
        }
        // Diagnostics borrow the original retained input rows before freeze.
        // No extra support/row-space scan occurs when this target is disabled.
        if tracing::enabled!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            tracing::Level::DEBUG
        ) {
            for (child, diagnostic) in self.collector.first_outside_fit_diagnostics() {
                let owner = &self.declaration.scopes[child].owner;
                tracing::debug!(
                    target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                    generation = self.generation,
                    phase = ?phase(self.phase),
                    child,
                    rows = owner.rows,
                    role = ?owner.role,
                    product = ?owner.product,
                    input_domain_prefix = %format_args!("{:016x}", diagnostic.input_domain_prefix),
                    support_axes = diagnostic.support_axes,
                    completion_support_offset = ?diagnostic.completion_support_offset,
                    reason = ?diagnostic.reason,
                    "Automatic first outside Fit support"
                );
            }
        }
        let freeze = self.collector.freeze(now).map_err(error)?;
        // Keep the original numerical decisions visible in MemoryOnly mode.
        // Emit each already bounded child separately: a full owner catalogue
        // on one line can exceed the serving guard's audit-line byte bound.
        if tracing::enabled!(
            target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
            tracing::Level::DEBUG
        ) {
            if let StructuredServiceRecordV6::PhaseFreeze {
                phase,
                frozen_at_ns,
                children,
                ..
            } = &freeze
            {
                tracing::debug!(
                    target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                    generation = self.generation,
                    ?phase,
                    frozen_at_ns,
                    child_count = children.len(),
                    "Automatic numerical phase freeze"
                );
                for child in children {
                    tracing::debug!(
                        target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                        generation = self.generation,
                        ?phase,
                        ?child,
                        "Automatic numerical phase child freeze"
                    );
                }
            }
        }
        self.write_diagnostic(live, &freeze);
        // Failed children cannot become fitted/calibrated again in this
        // collector. Do not spend another complete offered population on a
        // generation with no remaining candidate. Its original failed source
        // is still closed by the caller, with the full phase receipt retained.
        let all_children_failed = matches!(
            &freeze,
            StructuredServiceRecordV6::PhaseFreeze { children, .. }
                if !children.is_empty() && children.iter().all(|child| child.failure.is_some())
        );
        let shadow = std::mem::replace(
            &mut self.shadow,
            super::shadow::ShadowDiscovery::open(self.discovery_policy, 0, cutoff),
        )
        .freeze(window.audit(), now, cutoff);
        if all_children_failed {
            return Ok(PhaseCompletion::AllChildrenFailed(shadow));
        }
        // The next shadow uses only its own phase, never a favorable subset or
        // unfinished prefix of this window. There is at most one live shadow.
        drop(shadow);
        self.phase += 1;
        if self.phase < 3 {
            let at = clock
                .now_ns()
                .filter(|at| *at >= now)
                .ok_or_else(|| error("phase clock moved backwards"))?;
            self.record(
                live,
                &StructuredServiceRecordV6::PhaseOpen {
                    phase: phase(self.phase),
                    opened_at_ns: at,
                    fifo_cutoff: cutoff,
                },
            )?;
            if let Some(source) = &mut self.source {
                source.next_phase();
            }
            self.shadow = super::shadow::ShadowDiscovery::open(
                self.discovery_policy,
                self.declaration.phase_offered_waves[self.phase],
                cutoff,
            );
        } else {
            let closing = ExportClockReading::closing(clock).map_err(error)?;
            self.record(
                live,
                &StructuredServiceRecordV6::Footer {
                    offered: self.collector.offered(),
                    accepted_fifo_cutoff: self.collector.last_fifo(),
                    closing: paired(closing),
                    failure: None,
                },
            )?;
            if let Some(source) = &mut self.source {
                source.footer();
            }
        }
        // Keep the original last population until all closing operations pass;
        // failures retain it alongside the failed private ticket ledger.
        if self.phase < 3 {
            *collected = Collected::default();
        }
        Ok(PhaseCompletion::Continue)
    }

    pub(super) fn activate(
        mut self,
        live: &LiveCalibration,
        clock: &dyn CostObservationClock,
        import: &ferrum_types::SloCostProfileImportConfig,
    ) -> Result<
        (
            publication::Publication,
            Option<super::super::super::profile_export::PublishedFile>,
        ),
        FerrumError,
    > {
        let result = (|| {
            let mut published = self
                .source
                .as_mut()
                .and_then(|source| match source.publish() {
                    Ok(receipt) => Some(receipt),
                    Err(reason) => {
                        source.note_failure(live, self.generation, &reason.to_string());
                        live.record_automatic_diagnostic_failure(
                            self.generation,
                            super::source::DiagnosticFailureStage::Publish,
                            &reason,
                        );
                        None
                    }
                });
            if self.collector.qualified_children() == 0 {
                return Err(error("complete numerical population qualified no owners"));
            }
            let now = ExportClockReading::closing(clock).map_err(error)?;
            let limits = profile::load_limits(import);
            let imported = self
                .collector
                .activate_same_process_memory(paired(now), &limits)
                .map_err(error)?;
            if published.as_ref().is_some_and(|source| {
                source.bytes != imported.source_bytes || source.digest != imported.source_sha256
            }) {
                let reason = "diagnostic source receipt differs from the original sealed collector";
                live.record_automatic_diagnostic_failure(
                    self.generation,
                    super::source::DiagnosticFailureStage::ReceiptMismatch,
                    &reason,
                );
                if let Some(source) = &self.source {
                    source.note_failure(live, self.generation, reason);
                }
                published = None;
            }
            let (children, receipt) =
                profile::EngineCostSnapshot::live_service_catalog_with_domain(
                    None,
                    imported,
                    live.workload_domain(),
                )?;
            Ok((publication::Publication { children, receipt }, published))
        })();
        if let (Err(reason), Some(source)) = (&result, &self.source) {
            source.note_failure(live, self.generation, &reason.to_string());
        }
        result
    }

    pub(super) fn persist_failure(
        &mut self,
        live: &LiveCalibration,
        window: &tickets::Window,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        reason: &str,
    ) {
        if let Some(source) = &mut self.source {
            source.fail(live, &self.declaration, window, clock, cutoff, reason);
        }
    }
}
