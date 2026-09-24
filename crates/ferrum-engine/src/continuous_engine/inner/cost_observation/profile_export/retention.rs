//! Same-entry retention and sizing, performed only by the training worker.
use super::*;

impl Retained {
    pub(super) fn observation_record(&self) -> Option<RawRecord<'_>> {
        let row = self.observation.as_ref()?;
        Some(RawRecord::Observation {
            accepted_ordinal: self.accepted_ordinal,
            observed_at_monotonic_ns: row.observed_at_monotonic_ns,
            fingerprint: &row.fingerprint,
            training: row.training,
            pre_update_prediction: row.pre_update_prediction,
            sample: &row.sample,
        })
    }
    pub(super) fn stages_record(&self) -> Option<RawRecord<'_>> {
        Some(RawRecord::HostStagesV1 {
            accepted_ordinal: self.accepted_ordinal,
            source_record: self
                .observation
                .as_ref()
                .map(|row| row.sample.source_record),
            legacy_rejection: self.legacy_rejection,
            evidence: ExportEvidence::new(
                self.stages.as_deref()?,
                self.host_content
                    .as_ref()
                    .is_some_and(|row| row.row_multiset),
            ),
        })
    }
    pub(super) fn host_content_record(&self) -> Option<RawRecord<'_>> {
        let row = self.host_content.as_ref()?;
        if row.row_multiset {
            Some(RawRecord::HostRowMultisetTrainingV2 {
                accepted_ordinal: self.accepted_ordinal,
                host_source_record: row.host_source_record,
                evaluation: row.evaluation,
                sample: row.sample.as_ref().map(|sample| match sample {
                    RetainedHostSample::V4(sample) => sample,
                    RetainedHostSample::V3(_) => unreachable!("validated export mode"),
                }),
            })
        } else {
            Some(RawRecord::HostContentTrainingV1 {
                accepted_ordinal: self.accepted_ordinal,
                host_source_record: row.host_source_record,
                evaluation: row.evaluation,
                sample: row.sample.as_ref().map(|sample| match sample {
                    RetainedHostSample::V3(sample) => sample,
                    RetainedHostSample::V4(_) => unreachable!("validated export mode"),
                }),
            })
        }
    }
}

#[derive(Clone, Copy)]
enum Limit {
    Entries,
    Rows,
    Bytes,
}

impl ProfileExporter {
    fn drop_entry(&mut self, limit: Limit, observation: bool, stages: bool) {
        let c = &mut self.counts;
        if observation {
            let counter = match limit {
                Limit::Entries => &mut c.dropped_sample_limit,
                Limit::Rows => &mut c.dropped_shape_row_limit,
                Limit::Bytes => &mut c.dropped_file_byte_limit,
            };
            audit::add_one(counter, &mut c.counter_exhausted);
        }
        if stages {
            let counter = match limit {
                Limit::Entries => &mut c.host_stages_dropped_sample_limit,
                Limit::Rows => &mut c.host_stages_dropped_shape_row_limit,
                Limit::Bytes => &mut c.host_stages_dropped_file_byte_limit,
            };
            audit::add_one(counter, &mut c.counter_exhausted);
        }
    }

    #[cfg(test)]
    pub(super) fn record_entry(
        &mut self,
        accepted_ordinal: u64,
        observation: Option<(
            &model::WaveCostObservation,
            TrainingDisposition,
            PreUpdatePrediction,
        )>,
        stages: Option<Arc<HostStageEvidenceV1>>,
        legacy_rejection: Option<CostCallRejection>,
    ) -> Result<(), ExportError> {
        self.record_entry_with_host(
            accepted_ordinal,
            observation,
            stages,
            legacy_rejection,
            None,
        )
    }

    pub(super) fn record_entry_with_host(
        &mut self,
        accepted_ordinal: u64,
        observation: Option<(
            &model::WaveCostObservation,
            TrainingDisposition,
            PreUpdatePrediction,
        )>,
        stages: Option<Arc<HostStageEvidenceV1>>,
        legacy_rejection: Option<CostCallRejection>,
        host: Option<(HostContentEvaluation, Option<&model::WaveCostObservation>)>,
    ) -> Result<(), ExportError> {
        if observation.is_none() && stages.is_none() {
            return Err(ExportError::Source("empty evidence entry"));
        }
        if observation.is_some() && legacy_rejection.is_some() {
            return Err(ExportError::Source(
                "training entry has a legacy call rejection",
            ));
        }
        if self.last_accepted_ordinal.checked_add(1) != Some(accepted_ordinal) {
            return Err(ExportError::Source(
                "export accepted-evidence sequence changed",
            ));
        }
        self.last_accepted_ordinal = accepted_ordinal;
        audit::add_one(
            &mut self.counts.received_entries,
            &mut self.counts.counter_exhausted,
        );
        if stages.is_some() {
            audit::add_one(
                &mut self.counts.received_host_stages,
                &mut self.counts.counter_exhausted,
            );
        }
        if host.is_some() {
            audit::add_one(
                &mut self.counts.received_host_content_evaluations,
                &mut self.counts.counter_exhausted,
            );
        }
        let observation = if let Some((sample, training, prediction)) = observation {
            let next = self
                .counts
                .received_observations
                .checked_add(1)
                .ok_or_else(|| {
                    self.counts.counter_exhausted = true;
                    ExportError::Source("observation ordinal exhausted")
                })?;
            self.counts.received_observations = next;
            let wave = audit::wave_index(sample.actual_shape.kind);
            audit::add_one(
                &mut self.counts.received_by_wave[wave],
                &mut self.counts.counter_exhausted,
            );
            if !training.recorded() {
                audit::add_one(
                    &mut self.counts.trainer_not_recorded,
                    &mut self.counts.counter_exhausted,
                );
            }
            if sample.outcome != model::WaveObservationOutcome::Completed {
                audit::add_one(
                    &mut self.counts.non_completed,
                    &mut self.counts.counter_exhausted,
                );
                None
            } else {
                if training.recorded() && sample.fingerprint != self.plan.fingerprint {
                    return Err(ExportError::Source(
                        "observation fingerprint changed during capture",
                    ));
                }
                Some((sample, training, prediction, next - 1))
            }
        } else {
            None
        };
        if observation.is_none() && stages.is_none() {
            return Ok(());
        }
        let has_observation = observation.is_some();
        let has_stages = stages.is_some();
        if self.samples.len() >= self.plan.options.max_samples.get() {
            self.drop_entry(Limit::Entries, has_observation, has_stages);
            return Ok(());
        }
        let stage_rows = stages
            .as_ref()
            .map_or(Some(0), |stages| stages.retained_rows());
        let sample_rows = observation.as_ref().map_or(Some(0), |(sample, _, _, _)| {
            let shape = &sample.actual_shape;
            let work = shape
                .decode_kv_tokens
                .len()
                .checked_add(shape.prefill_chunks.len())?;
            let numeric = shape
                .numeric_features
                .as_ref()
                .map_or(0, |features| features.rows.len());
            if work > 1024 || numeric > 1024 {
                return None;
            }
            work.checked_add(numeric).map(|rows| rows.max(1))
        });
        // Check the additional profile copy before allocating it, then charge
        // the resulting Vec capacities again below. The source Arc is already
        // charged independently and cannot lend its row budget to this copy.
        let host_copy_rows = host
            .and_then(|(_, sample)| sample)
            .map_or(Some(0), |sample| {
                let shape = &sample.actual_shape;
                shape
                    .decode_kv_tokens
                    .len()
                    .checked_add(shape.prefill_chunks.len())?
                    .checked_add(
                        shape
                            .numeric_features
                            .as_ref()
                            .map_or(0, |features| features.rows.len()),
                    )?
                    .checked_add(if Self::row_multiset_mode(&self.plan) {
                        shape
                            .row_multiset_features
                            .as_ref()
                            .map_or(0, |features| features.rows.len())
                    } else {
                        0
                    })
            });
        if stage_rows
            .zip(sample_rows)
            .and_then(|(a, b)| a.checked_add(b))
            .zip(host_copy_rows)
            .and_then(|(a, b)| a.checked_add(b))
            .and_then(|rows| self.rows.checked_add(rows))
            .is_none_or(|rows| rows > self.plan.options.max_total_shape_rows.get())
        {
            self.drop_entry(Limit::Rows, has_observation, has_stages);
            return Ok(());
        }
        let host_content = host
            .map(|(evaluation, sample)| {
                host_content::retain(
                    &self.plan,
                    accepted_ordinal,
                    has_stages.then(|| self.counts.received_host_stages - 1),
                    evaluation,
                    sample,
                )
            })
            .transpose()?;
        let observation = observation
            .map(
                |(observation, training, pre_update_prediction, source_record)| {
                    let measured_unix_ns = observation
                        .observed_at_ns
                        .checked_sub(self.plan.opening.monotonic_ns)
                        .and_then(|elapsed| self.plan.opening.wall_unix_ns.checked_add(elapsed))
                        .ok_or(ExportError::Clock(
                            "receipt predates anchor or wall timestamp overflow",
                        ))?;
                    Ok::<_, ExportError>(RetainedObservation {
                        observed_at_monotonic_ns: observation.observed_at_ns,
                        fingerprint: profile::ProfileFingerprint::from(&observation.fingerprint),
                        training,
                        pre_update_prediction,
                        sample: sample_from_observation(
                            observation,
                            source_record,
                            measured_unix_ns,
                        )?,
                    })
                },
            )
            .transpose()?;
        // Charge actual allocated wire-row capacities, as well as the retained
        // host Arc's original capacities. Arc clone does not refresh its clock.
        let sample_rows = observation.as_ref().map_or(Some(0), |row| {
            let shape = &row.sample.shape;
            shape
                .exact
                .decode_kv_tokens
                .capacity()
                .checked_add(shape.exact.prefill_chunks.capacity())?
                .checked_add(
                    shape
                        .numeric_features
                        .as_ref()
                        .map_or(0, |f| f.rows.capacity()),
                )
                .map(|rows| rows.max(1))
        });
        let host_rows = host_content
            .as_ref()
            .map_or(Some(0), host_content::retained_rows);
        let Some(total_rows) = stage_rows
            .zip(sample_rows)
            .and_then(|(a, b)| a.checked_add(b))
            .zip(host_rows)
            .and_then(|(a, b)| a.checked_add(b))
            .and_then(|rows| self.rows.checked_add(rows))
            .filter(|rows| *rows <= self.plan.options.max_total_shape_rows.get())
        else {
            self.drop_entry(Limit::Rows, has_observation, has_stages);
            return Ok(());
        };
        let entry = Retained {
            accepted_ordinal,
            observation,
            stages,
            legacy_rejection,
            host_content,
        };
        let mut bytes = 0u64;
        for record in [
            entry.observation_record(),
            entry.stages_record(),
            entry.host_content_record(),
        ]
        .into_iter()
        .flatten()
        {
            bytes = bytes
                .checked_add(json_size(&record)?)
                .and_then(|n| n.checked_add(1))
                .ok_or(ExportError::Source("record size overflow"))?;
        }
        let Some(body_bytes) = self
            .body_bytes
            .checked_add(bytes)
            .filter(|n| *n <= self.max_body_bytes)
        else {
            self.drop_entry(Limit::Bytes, has_observation, has_stages);
            return Ok(());
        };
        self.rows = total_rows;
        self.body_bytes = body_bytes;
        audit::add_one(
            &mut self.counts.raw_retained_entries,
            &mut self.counts.counter_exhausted,
        );
        if has_stages {
            audit::add_one(
                &mut self.counts.raw_retained_host_stages,
                &mut self.counts.counter_exhausted,
            );
        }
        if let Some(row) = &entry.host_content {
            audit::add_one(
                &mut self.counts.raw_retained_host_content_evaluations,
                &mut self.counts.counter_exhausted,
            );
            if row.evaluation.training.recorded() && row.sample.is_some() {
                audit::add_one(
                    &mut self.counts.retained_host_content_samples,
                    &mut self.counts.counter_exhausted,
                );
            }
        }
        if let Some(row) = &entry.observation {
            let wave = audit::wave_index(model::WaveKind::from(row.sample.shape.exact.kind));
            audit::add_one(
                &mut self.counts.raw_retained_observations,
                &mut self.counts.counter_exhausted,
            );
            audit::add_one(
                &mut self.counts.raw_retained_by_wave[wave],
                &mut self.counts.counter_exhausted,
            );
            if row.training.recorded() {
                audit::add_one(
                    &mut self.counts.retained_samples,
                    &mut self.counts.counter_exhausted,
                );
                audit::add_one(
                    &mut self.counts.profile_retained_by_wave[wave],
                    &mut self.counts.counter_exhausted,
                );
            }
        }
        self.samples.push(entry);
        Ok(())
    }
}
