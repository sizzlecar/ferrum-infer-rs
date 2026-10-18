//! Bind backend observations only after the original submission is armed.
use super::{foundation::invalid_operation, work_facts::PreparedSubmissionWork};
use crate::*;

struct ExpectedCompute {
    command_index: u32,
    node_index: Option<u32>,
    /// Direct reusable commands expand to the declared original node range.
    replay_ordinal: Option<u32>,
}

pub(super) struct PreparedDeviceFacts {
    batch_command_count: u32,
    compute: Vec<ExpectedCompute>,
}

impl PreparedDeviceFacts {
    pub(super) fn capture<C>(
        capture: ExecutionFactsCapture,
        commands: &DeviceCommandBatch<C>,
        replay: Option<&DeviceReusableExecutionProgram>,
    ) -> Result<Option<Self>, VNextError> {
        if capture == ExecutionFactsCapture::Off {
            return Ok(None);
        }
        let batch_command_count = u32::try_from(commands.entries().len())
            .map_err(|_| invalid_operation("device facts batch exceeds u32"))?;
        let mut compute = Vec::new();
        for (index, entry) in commands.entries().iter().enumerate() {
            if entry.phase() != DeviceCommandPhase::Compute {
                continue;
            }
            let segment = replay.and_then(|program| {
                program
                    .segments()
                    .iter()
                    .find(|segment| entry.node_index() == Some(segment.start_node_index()))
            });
            if let Some(segment) = segment {
                if segment.logical_command_count()
                    != segment.end_node_index() - segment.start_node_index()
                {
                    return Err(invalid_operation(
                        "device facts reusable segment is not one command per node",
                    ));
                }
                for (ordinal, node) in
                    (segment.start_node_index()..segment.end_node_index()).enumerate()
                {
                    compute.push(ExpectedCompute {
                        command_index: index as u32,
                        node_index: Some(node),
                        replay_ordinal: Some(ordinal as u32),
                    });
                }
            } else {
                compute.push(ExpectedCompute {
                    command_index: index as u32,
                    node_index: entry.node_index(),
                    replay_ordinal: None,
                });
            }
        }
        Ok(Some(Self {
            batch_command_count,
            compute,
        }))
    }

    fn bind(
        self,
        device: DeviceSubmissionExecutionFacts,
        work: &BoundSubmissionWork,
    ) -> Result<BoundSubmissionDeviceFacts, VNextError> {
        if device.batch_command_count() != self.batch_command_count
            || device.compute_commands().len() != self.compute.len()
        {
            return Err(invalid_operation(
                "device facts compute coverage differs from the submitted command batch",
            ));
        }
        let mut projection_rows = Vec::new();
        let mut unavailable = 0_u32;
        let mut adaptive_replay_next = None;
        for (expected, command) in self.compute.iter().zip(device.compute_commands()) {
            if command.submission_command_index != expected.command_index
                || command.node_index != expected.node_index
            {
                return Err(invalid_operation(
                    "device facts command/node order differs from the submitted batch",
                ));
            }
            match command.path {
                DeviceExecutionPath::Eager
                    if command.replay_command_ordinal.is_some()
                        || expected.replay_ordinal.is_some() =>
                {
                    return Err(invalid_operation(
                        "eager device facts carry replay coordinates",
                    ));
                }
                DeviceExecutionPath::Replayed
                    if command.replay_command_ordinal.is_none()
                        || expected.replay_ordinal.is_some_and(|ordinal| {
                            command.replay_command_ordinal != Some(ordinal)
                        }) =>
                {
                    return Err(invalid_operation(
                        "device facts replay ordinal differs from the submitted segment",
                    ));
                }
                _ => {}
            }
            match (
                expected.replay_ordinal,
                command.path,
                command.replay_command_ordinal,
            ) {
                (None, DeviceExecutionPath::Replayed, Some(ordinal)) => {
                    validate_adaptive_replay_ordinal(
                        &mut adaptive_replay_next,
                        command.submission_command_index,
                        ordinal,
                    )?;
                }
                _ => adaptive_replay_next = None,
            }
            match &command.coverage {
                ProjectionCommandCoverage::Unavailable(reason) => {
                    if *reason == DeviceExecutionFactsUnavailableReason::CachedTemplateUnavailable
                        && command.path != DeviceExecutionPath::Replayed
                    {
                        return Err(invalid_operation(
                            "eager device facts claim a missing cached template",
                        ));
                    }
                    unavailable += 1;
                }
                ProjectionCommandCoverage::Covered(templates) => {
                    if templates.is_empty() {
                        return Err(invalid_operation(
                            "covered compute command has no projection",
                        ));
                    }
                    let node_index = command
                        .node_index
                        .ok_or_else(|| invalid_operation("projection has no submitted node"))?;
                    let node = work
                        .nodes()
                        .get(node_index as usize)
                        .filter(|node| node.node_index() == node_index)
                        .ok_or_else(|| {
                            invalid_operation(
                                "projection node is absent from current submitted work",
                            )
                        })?;
                    for (ordinal, template) in templates.iter().enumerate() {
                        if template.ordinal as usize != ordinal {
                            return Err(invalid_operation(
                                "projection ordinals are not in execution order",
                            ));
                        }
                        validate_projection(template)?;
                        let mut rows = Vec::with_capacity(template.rows.len());
                        let mut physical_end = 0;
                        for selector in &template.rows {
                            if selector.physical_rows.start != physical_end
                                || selector.physical_rows.end
                                    != physical_end.checked_add(1).ok_or_else(|| {
                                        invalid_operation("projection row overflow")
                                    })?
                                || rows.iter().any(|row: &ResolvedProjectionRow| {
                                    row.participant_index == selector.participant_index
                                })
                            {
                                return Err(invalid_operation("projection physical rows are not an exact ordered participant selection"));
                            }
                            let participant = node
                                .participants()
                                .get(selector.participant_index as usize)
                                .ok_or_else(|| {
                                    invalid_operation(
                                        "projection selector exceeds current node participants",
                                    )
                                })?;
                            let source = participant.source_token_range();
                            let packed = participant.immediate_token_range();
                            if source.is_empty() || packed.is_empty() {
                                return Err(invalid_operation(
                                    "projection selects an empty token window",
                                ));
                            }
                            let ParticipantTokenSelector::LastSourceToken = selector.source;
                            rows.push(ResolvedProjectionRow {
                                participant_index: selector.participant_index,
                                source_token_range: source.end - 1..source.end,
                                packed_token_range: packed.end - 1..packed.end,
                                physical_rows: selector.physical_rows.clone(),
                            });
                            physical_end = selector.physical_rows.end;
                        }
                        if physical_end != template.dimensions.m {
                            return Err(invalid_operation(
                                "projection selectors do not cover actual matrix rows",
                            ));
                        }
                        projection_rows.push(BoundProjectionRows {
                            submission_command_index: command.submission_command_index,
                            replay_command_ordinal: command.replay_command_ordinal,
                            node_index,
                            projection_ordinal: template.ordinal,
                            rows: rows.into_boxed_slice(),
                        });
                    }
                }
            }
        }
        let count = u32::try_from(self.compute.len())
            .map_err(|_| invalid_operation("device facts compute count exceeds u32"))?;
        Ok(BoundSubmissionDeviceFacts::new(
            work.submission().clone(),
            device,
            projection_rows.into_boxed_slice(),
            DeviceExecutionFactsCoverage::ComputeCommands {
                described_compute_commands: count - unavailable,
                unavailable_compute_commands: unavailable,
            },
        ))
    }
}

// A candidate consists of contiguous compute commands. Zero can also start a
// distinct adjacent candidate; these coordinates do not identify a cache entry.
fn validate_adaptive_replay_ordinal(
    next: &mut Option<(u32, u32)>,
    command_index: u32,
    ordinal: u32,
) -> Result<(), VNextError> {
    if ordinal != 0 && *next != Some((command_index, ordinal)) {
        return Err(invalid_operation(
            "adaptive replay command ordinals are not contiguous",
        ));
    }
    *next = command_index.checked_add(1).zip(ordinal.checked_add(1));
    Ok(())
}

fn validate_launch(launch: &ProjectionLaunchGeometry) -> Result<(), VNextError> {
    if launch.grid.contains(&0) || launch.block.contains(&0) {
        return Err(invalid_operation(
            "projection launch has an empty dimension",
        ));
    }
    Ok(())
}

fn validate_projection(template: &ProjectionInvocationTemplate) -> Result<(), VNextError> {
    let d = template.dimensions;
    let l = template.layout;
    if d.m == 0
        || d.k == 0
        || d.n == 0
        || l.input_stride < d.k
        || l.output_offset
            .checked_add(d.n)
            .is_none_or(|end| end > l.output_stride)
    {
        return Err(invalid_operation(
            "projection dimensions exceed their actual layout",
        ));
    }
    match &template.arithmetic {
        ProjectionArithmetic::Direct {
            row_tile, launch, ..
        } => {
            if *row_tile == 0 {
                return Err(invalid_operation("projection row tile is empty"));
            }
            validate_launch(launch)?;
        }
        ProjectionArithmetic::PackedQuantized { geometry, .. } => {
            if geometry.padded_k < d.k
                || geometry.padded_n < d.n
                || geometry.row_tile == 0
                || geometry.column_tile == 0
                || geometry.blocks == 0
                || geometry.threads == 0
                || geometry.packed_bytes == 0
                || geometry.fixup != (geometry.fixup_bytes != 0)
                || geometry
                    .packed_bytes
                    .checked_add(geometry.fixup_bytes)
                    .is_none_or(|minimum| geometry.workspace_bytes < minimum)
            {
                return Err(invalid_operation(
                    "packed projection geometry or workspace cannot cover the matrix",
                ));
            }
        }
    }
    for transform in &template.input_transforms {
        transform.spec.validate(u64::from(d.k))?;
        if !matches!(
            transform.spec.application,
            HadamardApplication::BeforeMatmul { .. }
        ) {
            return Err(invalid_operation(
                "projection input transform is not before-matmul",
            ));
        }
        validate_launch(&transform.launch)?;
    }
    Ok(())
}

#[cfg(test)]
mod replay_coordinate_tests {
    use super::validate_adaptive_replay_ordinal;

    #[test]
    fn execution_device_facts_adaptive_replay_coordinate_contract() {
        let mut next = None;
        assert!(validate_adaptive_replay_ordinal(&mut next, 3, 99).is_err());
        validate_adaptive_replay_ordinal(&mut next, 3, 0).unwrap();
        validate_adaptive_replay_ordinal(&mut next, 4, 1).unwrap();
        // A separately captured adjacent candidate starts over at zero.
        validate_adaptive_replay_ordinal(&mut next, 5, 0).unwrap();
        validate_adaptive_replay_ordinal(&mut next, 6, 1).unwrap();
        assert!(validate_adaptive_replay_ordinal(&mut next, 7, 5).is_err());
        // Non-compute entries interrupt a candidate even though they are absent
        // from the compute-only coverage list.
        assert!(validate_adaptive_replay_ordinal(&mut next, 8, 2).is_err());
        validate_adaptive_replay_ordinal(&mut next, 8, 0).unwrap();
    }
}

/// Getters are isolated from each other. Even if one panics, the original fence
/// must be armed and the other successful attachment must remain observable.
pub(super) struct SubmissionAttachments {
    attribution: Result<Option<DeviceSubmissionAttribution>, VNextError>,
    device: Result<Option<DeviceSubmissionExecutionFacts>, VNextError>,
    panicked: bool,
}

impl SubmissionAttachments {
    pub(super) fn read<R: DeviceRuntime>(
        runtime: &R,
        fence: &R::Fence,
        capture: ExecutionFactsCapture,
    ) -> Self {
        let attribution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.submission_attribution(fence)
        }));
        let device = if capture == ExecutionFactsCapture::On {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.submission_execution_facts(fence).map_err(|error| {
                    invalid_operation(format!(
                        "device facts inspection failed after submission: {error}"
                    ))
                })
            }))
        } else {
            Ok(Ok(None))
        };
        let panicked = attribution.is_err() || device.is_err();
        Self {
            attribution: attribution
                .map_err(|_| invalid_operation("submission attribution panicked after submission")),
            device: device.unwrap_or_else(|_| {
                Err(invalid_operation(
                    "device facts inspection panicked after submission",
                ))
            }),
            panicked,
        }
    }

    pub(super) fn panicked(&self) -> bool {
        self.panicked
    }

    pub(super) fn bind(
        self,
        receipt: &SubmittedOperationReceipt,
        work: Option<PreparedSubmissionWork>,
        expected: Option<PreparedDeviceFacts>,
    ) -> (SubmissionObservation, Option<VNextError>) {
        let work = work.map(|work| work.bind(receipt));
        let mut failure = None;
        let attribution = match self.attribution.and_then(|device| {
            device
                .map(|device| {
                    BoundDeviceSubmissionAttribution::new(
                        receipt.batch_identity().clone(),
                        receipt.fingerprint().to_owned(),
                        device,
                    )
                })
                .transpose()
        }) {
            Ok(attribution) => attribution,
            Err(error) => {
                failure = Some(error);
                None
            }
        };
        let device = match self.device.and_then(|device| match device {
            None => Ok(None),
            Some(device) => match (expected, work.as_ref()) {
                (Some(expected), Some(work)) => expected.bind(device, work).map(Some),
                _ => Err(invalid_operation(
                    "device facts have no original captured work",
                )),
            },
        }) {
            Ok(device) => device,
            Err(error) => {
                if failure.is_none() {
                    failure = Some(error);
                }
                None
            }
        };
        (
            SubmissionObservation::new(work, attribution).with_device_facts(device),
            failure,
        )
    }
}
