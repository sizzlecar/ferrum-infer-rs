//! Stable role indices select compact per-request inputs across evaluations.
//! The live source retains original time rows; role tables are admitted copies.
use super::*;
use crate::fast_h3::FastH3ConditioningTable;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FastH3ConditioningRole {
    RetainedVision,
    RetainedAudio,
    VideoVision,
    VideoText,
    TargetAudio,
}
impl FastH3ConditioningRole {
    const ALL: [Self; 5] = [
        Self::RetainedVision,
        Self::RetainedAudio,
        Self::VideoVision,
        Self::VideoText,
        Self::TargetAudio,
    ];
    fn index(self) -> u64 {
        match self {
            Self::RetainedVision => 0,
            Self::RetainedAudio => 1,
            Self::VideoVision => 2,
            Self::VideoText => 3,
            Self::TargetAudio => 4,
        }
    }
    fn kind(self) -> FastH3SegmentKind {
        match self {
            Self::RetainedVision => FastH3SegmentKind::ReferenceImage,
            Self::RetainedAudio => FastH3SegmentKind::ReferenceAudio,
            Self::VideoVision => FastH3SegmentKind::TargetVideo,
            Self::VideoText => FastH3SegmentKind::Text,
            Self::TargetAudio => FastH3SegmentKind::TargetAudio,
        }
    }
    fn modality(self) -> u64 {
        match self {
            Self::RetainedAudio | Self::TargetAudio => 2,
            Self::VideoText => 1,
            _ => 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RoleRun {
    rows: u64,
    request: u64,
    role: FastH3ConditioningRole,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Routing {
    Scheduled {
        runs: Vec<Run>,
    },
    ConditioningRoles {
        requests: u64,
        time_rows: Vec<[usize; 5]>,
        runs: Vec<RoleRun>,
    },
}
impl Routing {
    pub(super) fn parameter_row_runs(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let (scheduled, roles) = match self {
            Self::Scheduled { runs } => (runs.as_slice(), &[][..]),
            Self::ConditioningRoles { runs, .. } => (&[][..], runs.as_slice()),
        };
        scheduled
            .iter()
            .map(|r| (r.rows, 3 * r.time_row as u64 + r.modality))
            .chain(
                roles
                    .iter()
                    .map(|r| (r.rows, 5 * r.request + r.role.index())),
            )
    }
    pub(super) fn table_rows(&self, times: usize) -> Result<u64, VNextError> {
        match self {
            Self::Scheduled { .. } => (times as u64).checked_mul(3),
            Self::ConditioningRoles { requests, .. } => requests.checked_mul(5),
        }
        .ok_or_else(|| invalid("table rows overflow"))
    }
    pub(super) fn validate(&self, rows: u64, times: usize) -> Result<(), VNextError> {
        let invalid_runs = match self {
            Self::Scheduled { runs } => runs
                .iter()
                .any(|r| r.rows == 0 || r.time_row >= times || r.modality >= 3),
            Self::ConditioningRoles {
                requests,
                time_rows,
                runs,
            } => {
                *requests == 0
                    || requests.checked_mul(5).is_none()
                    || time_rows.is_empty()
                    || time_rows.iter().flatten().any(|&t| t >= times)
                    || runs.iter().any(|r| r.rows == 0 || r.request >= *requests)
                    || runs.first().is_none_or(|r| r.request != 0)
                    || runs.last().is_none_or(|r| r.request + 1 != *requests)
                    || runs
                        .windows(2)
                        .any(|w| w[1].request < w[0].request || w[1].request > w[0].request + 1)
            }
        };
        let mut previous = None;
        let mut count = 0;
        let mut total = 0u64;
        for (n, row) in self.parameter_row_runs() {
            if previous == Some(row) {
                return Err(invalid("routing must use canonical runs"));
            }
            total = total
                .checked_add(n)
                .ok_or_else(|| invalid("routing extent overflow"))?;
            count += 1;
            previous = Some(row);
        }
        if invalid_runs || total != rows || count == 0 || count > ROW_SELECTED_BF16_MAX_RUNS {
            return Err(invalid("invalid bounded time/modality routing"));
        }
        Ok(())
    }
}
fn schedule_rows(schedule: &FastH3ConditioningSchedule) -> Result<Vec<[usize; 5]>, VNextError> {
    (0..schedule.evaluation_count())
        .map(|evaluation| {
            let mut rows = [0; 5];
            for (i, role) in FastH3ConditioningRole::ALL.into_iter().enumerate() {
                rows[i] = schedule
                    .segment_time_index(evaluation, role.kind())
                    .ok_or_else(|| {
                        invalid("role routing requires an all-modalities conditioning schedule")
                    })?;
            }
            Ok(rows)
        })
        .collect()
}

impl FastH3ModulationConfig {
    /// Unlike scheduled routing, this configuration has no current evaluation.
    /// Text vision tokens follow target video time, not retained-image time.
    pub fn from_conditioning_roles(
        package: &FastH3TransformerPackage,
        block: usize,
        stage: FastH3ModulationStage,
        layout: &FastH3PackedLayout,
        schedule: &FastH3ConditioningSchedule,
        requests: &[&[FastH3TextTagRun]],
        limits: FastH3ModulationLimits,
    ) -> Result<Self, VNextError> {
        let mut runs = Vec::<RoleRun>::new();
        for (request, tags) in requests.iter().enumerate() {
            // Reuse the original source/layout/tag validation before constructing
            // stable roles; equal times at step zero must not collapse roles.
            Self::from_package(package, block, stage, layout, schedule, 0, tags, limits)?;
            let mut push = |rows, role| -> Result<(), VNextError> {
                let request = u64::try_from(request).map_err(invalid)?;
                if let Some(last) = runs
                    .last_mut()
                    .filter(|r| r.request == request && r.role == role)
                {
                    last.rows = last
                        .rows
                        .checked_add(rows)
                        .ok_or_else(|| invalid("role run overflow"))?;
                } else {
                    if runs.len() == ROW_SELECTED_BF16_MAX_RUNS {
                        return Err(invalid("role routing exceeds run capacity"));
                    }
                    runs.push(RoleRun {
                        rows,
                        request,
                        role,
                    });
                }
                Ok(())
            };
            for segment in layout.segments() {
                let role = match segment.kind {
                    FastH3SegmentKind::Text => {
                        for tag in *tags {
                            push(
                                tag.rows,
                                match tag.tag {
                                    FastH3TextTokenTag::Vision => {
                                        FastH3ConditioningRole::VideoVision
                                    }
                                    FastH3TextTokenTag::Text => FastH3ConditioningRole::VideoText,
                                },
                            )?;
                        }
                        continue;
                    }
                    FastH3SegmentKind::ReferenceImage | FastH3SegmentKind::ReferenceVideo => {
                        FastH3ConditioningRole::RetainedVision
                    }
                    FastH3SegmentKind::ReferenceAudio => FastH3ConditioningRole::RetainedAudio,
                    FastH3SegmentKind::TargetVideo => FastH3ConditioningRole::VideoVision,
                    FastH3SegmentKind::TargetAudio => FastH3ConditioningRole::TargetAudio,
                };
                push(segment.packed_rows.len().try_into().map_err(invalid)?, role)?;
            }
        }
        let requests = u64::try_from(requests.len()).map_err(invalid)?;
        let c = Self {
            block,
            stage,
            rows: requests
                .checked_mul(layout.sequence_rows().try_into().map_err(invalid)?)
                .ok_or_else(|| invalid("request rows overflow"))?,
            width: package.config().hidden_size().try_into().map_err(invalid)?,
            times: schedule.timesteps().to_vec(),
            routing: Routing::ConditioningRoles {
                requests,
                time_rows: schedule_rows(schedule)?,
                runs,
            },
            epsilon: CanonicalRational::from_decimal_str(&package.config().norm_eps().to_string())?,
            gamma: gamma(package, block, stage)?,
        };
        c.validate(limits)?;
        Ok(c)
    }
    pub fn conditioning_request_count(&self) -> Option<u64> {
        match self.routing {
            Routing::ConditioningRoles { requests, .. } => Some(requests),
            _ => None,
        }
    }
    pub(crate) fn same_routing(&self, other: &Self) -> bool {
        self.routing == other.routing
    }

    /// Select exact BF16 rows from the original live Time result. One role table
    /// per request is independent of sequence length and is a normal managed
    /// InputPreprocess allocation. No model weights or plan attributes change.
    pub fn managed_role_table(
        &self,
        table: &FastH3ConditioningTable,
        schedule: &FastH3ConditioningSchedule,
        evaluations: &[usize],
        host: &HostMemoryAccount,
    ) -> Result<ManagedHostTensor, VNextError> {
        let Routing::ConditioningRoles {
            requests,
            time_rows,
            ..
        } = &self.routing
        else {
            return Err(invalid("role input requires conditioning-role routing"));
        };
        if table.configuration().block() != self.block
            || table.configuration().hidden_size() != self.width
            || evaluations.len() as u64 != *requests
            || !self
                .times
                .iter()
                .map(|v| v.to_bits())
                .eq(table.timesteps().iter().map(|v| v.to_bits()))
            || !self
                .times
                .iter()
                .map(|v| v.to_bits())
                .eq(schedule.timesteps().iter().map(|v| v.to_bits()))
            || *time_rows != schedule_rows(schedule)?
            || evaluations.iter().any(|&e| e >= time_rows.len())
        {
            return Err(invalid(
                "live table, schedule or request evaluation differs from role routing",
            ));
        }
        let spec = self.table_spec()?;
        let bytes = usize::try_from(spec.byte_len()?).map_err(invalid)?;
        let row_bytes = usize::try_from(
            self.width
                .checked_mul(12)
                .ok_or_else(|| invalid("role width overflow"))?,
        )
        .map_err(invalid)?;
        let mut output = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes)
            .map_err(invalid)?;
        let original = table.outputs()[1].bytes().bytes();
        for (request, &evaluation) in evaluations.iter().enumerate() {
            for role in FastH3ConditioningRole::ALL {
                let source_row =
                    3 * time_rows[evaluation][role.index() as usize] + role.modality() as usize;
                let from = source_row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| invalid("source row overflow"))?;
                let source = original
                    .get(
                        from..from
                            .checked_add(row_bytes)
                            .ok_or_else(|| invalid("source extent overflow"))?,
                    )
                    .ok_or_else(|| invalid("live table does not cover role source row"))?;
                let to = (5 * request + role.index() as usize) * row_bytes;
                output.as_mut()[to..to + row_bytes].copy_from_slice(source);
            }
        }
        ManagedHostTensor::new(
            id(FastH3ModulationFamily::TABLE_ID)?,
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout)?,
            output.freeze(),
        )
    }
}
