//! Compute-command descriptions from the actual enqueue path. This collector
//! owns no command, buffer, completion, or replay authority. Transport commands,
//! terminal copies, and status readbacks are outside this projection's scope.

use super::super::vnext_replay::CudaExecutableLaunch;
use super::*;
use ferrum_runtime::{
    DeviceCommandExecutionFacts, DeviceExecutionFactsUnavailableReason,
    DeviceReusableExecutionSegment, ProjectionCommandCoverage,
};

pub(super) struct CudaSubmissionComputeFacts {
    batch_command_count: u32,
    // Physical queue positions may change when fresh uploads coalesce. These
    // coordinates always name the original core batch's compute commands.
    coordinates: Vec<Option<(u32, Option<u32>)>>,
    compute_commands: Vec<DeviceCommandExecutionFacts>,
}

impl CudaSubmissionComputeFacts {
    pub(super) fn new(
        entries: &[(DeviceCommandPhase, Option<u32>, CudaDeviceCommand)],
        batch_command_count: u32,
    ) -> Self {
        Self {
            batch_command_count,
            coordinates: entries
                .iter()
                .enumerate()
                .map(|(index, (phase, node, _))| {
                    (*phase == DeviceCommandPhase::Compute).then_some((index as u32, *node))
                })
                .collect(),
            compute_commands: Vec::new(),
        }
    }

    pub(super) fn after_upload_coalescing(
        &mut self,
        entries: &[(DeviceCommandPhase, Option<u32>, CudaDeviceCommand)],
    ) {
        // The original coalescer only compacts fresh eager uploads, and keeps
        // every compute command in order. Never change that algorithm for facts.
        let mut original_compute = self.coordinates.iter().copied().flatten();
        self.coordinates = entries
            .iter()
            .map(|(phase, _, _)| {
                (*phase == DeviceCommandPhase::Compute)
                    .then(|| original_compute.next())
                    .flatten()
            })
            .collect();
    }

    fn coordinate(&self, index: usize) -> (u32, Option<u32>) {
        self.coordinates
            .get(index)
            .copied()
            .flatten()
            // Preserve an invalid coordinate for the core's post-arm check.
            .unwrap_or((self.batch_command_count, None))
    }

    pub(super) fn eager(&mut self, index: usize, command: &CudaDeviceCommand) {
        let (submission_command_index, node_index) = self.coordinate(index);
        self.compute_commands.push(DeviceCommandExecutionFacts {
            submission_command_index,
            replay_command_ordinal: None,
            node_index,
            path: DeviceExecutionPath::Eager,
            coverage: command.execution_facts().map_or(
                ProjectionCommandCoverage::Unavailable(
                    DeviceExecutionFactsUnavailableReason::UnsupportedProducer,
                ),
                ProjectionCommandCoverage::Covered,
            ),
        });
    }

    pub(super) fn adaptive_replay(
        &mut self,
        start: usize,
        end: usize,
        launch: &CudaExecutableLaunch,
    ) {
        let templates = launch.execution_facts();
        // If an entry's metadata ever drifts, preserve its actual length for
        // the core's post-arm validation rather than silently repairing it.
        let count = templates.map_or(end - start, |slots| slots.len());
        for ordinal in 0..count {
            let index = start + ordinal;
            let (submission_command_index, node_index) = self.coordinate(index);
            self.compute_commands.push(DeviceCommandExecutionFacts {
                submission_command_index,
                replay_command_ordinal: Some(ordinal as u32),
                node_index,
                path: DeviceExecutionPath::Replayed,
                coverage: replay_coverage(templates.map(|slots| &slots[ordinal])),
            });
        }
    }

    pub(super) fn resident_replay(
        &mut self,
        index: usize,
        segment: &DeviceReusableExecutionSegment,
        launch: &CudaExecutableLaunch,
    ) {
        let templates = launch.execution_facts();
        let (submission_command_index, _) = self.coordinate(index);
        let count = templates.map_or(segment.logical_command_count() as usize, |slots| {
            slots.len()
        });
        for ordinal in 0..count {
            self.compute_commands.push(DeviceCommandExecutionFacts {
                submission_command_index,
                replay_command_ordinal: Some(ordinal as u32),
                node_index: segment.start_node_index().checked_add(ordinal as u32),
                path: DeviceExecutionPath::Replayed,
                coverage: replay_coverage(templates.map(|slots| &slots[ordinal])),
            });
        }
    }

    pub(super) fn finish(self) -> DeviceSubmissionExecutionFacts {
        DeviceSubmissionExecutionFacts::new(self.batch_command_count, self.compute_commands)
    }
}

fn replay_coverage(
    original: Option<&Option<Arc<[ProjectionInvocationTemplate]>>>,
) -> ProjectionCommandCoverage {
    match original {
        Some(Some(templates)) => ProjectionCommandCoverage::Covered(Arc::clone(templates)),
        Some(None) => ProjectionCommandCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::UnsupportedProducer,
        ),
        None => ProjectionCommandCoverage::Unavailable(
            DeviceExecutionFactsUnavailableReason::CachedTemplateUnavailable,
        ),
    }
}
