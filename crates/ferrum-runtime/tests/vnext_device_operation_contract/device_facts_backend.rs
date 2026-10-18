//! Controlled provider facts on the real TestRuntime submit/fence route. These
//! exercise Core's mapping and post-submit ownership, not a numerical kernel.
use super::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum DeviceFactsFault {
    #[default]
    None,
    MissingCompute,
    WrongNode,
    WrongReplayOrdinal,
    AdaptiveReplayFirstOrdinal,
    AdaptiveReplaySkippedOrdinal,
    GetterError,
    GetterPanic,
}

pub(crate) fn test_device_facts(
    entries: &[DeviceCommandEntry<TestCommand>],
    replays: &[DeviceReusableExecutionInvocation],
    capture: ExecutionFactsCapture,
    templates: Option<&Arc<[ProjectionInvocationTemplate]>>,
    fault: DeviceFactsFault,
) -> Option<DeviceSubmissionExecutionFacts> {
    if capture == ExecutionFactsCapture::Off {
        return None;
    }
    let templates = templates?;
    let mut commands = Vec::new();
    let mut replays = replays.iter();
    for (index, entry) in entries.iter().enumerate() {
        if entry.phase() != DeviceCommandPhase::Compute {
            continue;
        }
        if *entry.command() == TestCommand::ReusableExecution {
            let segment = replays
                .next()
                .expect("actual replay command must carry original invocation")
                .segment();
            for (ordinal, node) in
                (segment.start_node_index()..segment.end_node_index()).enumerate()
            {
                commands.push(DeviceCommandExecutionFacts {
                    submission_command_index: index as u32,
                    replay_command_ordinal: Some(ordinal as u32),
                    node_index: Some(node),
                    path: DeviceExecutionPath::Replayed,
                    coverage: ProjectionCommandCoverage::Covered(Arc::clone(templates)),
                });
            }
        } else {
            commands.push(DeviceCommandExecutionFacts {
                submission_command_index: index as u32,
                replay_command_ordinal: None,
                node_index: entry.node_index(),
                path: DeviceExecutionPath::Eager,
                coverage: ProjectionCommandCoverage::Covered(Arc::clone(templates)),
            });
        }
    }
    match fault {
        DeviceFactsFault::MissingCompute => {
            drop(commands.pop());
        }
        DeviceFactsFault::WrongNode => {
            commands[0].node_index = Some(u32::MAX);
        }
        DeviceFactsFault::WrongReplayOrdinal => {
            commands[0].replay_command_ordinal = Some(u32::MAX);
        }
        DeviceFactsFault::AdaptiveReplayFirstOrdinal => {
            // Deliberately malformed metadata after actual eager execution.
            // This is not evidence that the test backend performed a replay.
            commands[0].path = DeviceExecutionPath::Replayed;
            commands[0].replay_command_ordinal = Some(99);
        }
        DeviceFactsFault::AdaptiveReplaySkippedOrdinal => {
            assert!(
                commands.len() >= 2,
                "skip injection requires two actual compute entries"
            );
            commands[0].path = DeviceExecutionPath::Replayed;
            commands[0].replay_command_ordinal = Some(0);
            commands[1].path = DeviceExecutionPath::Replayed;
            commands[1].replay_command_ordinal = Some(5);
        }
        _ => {}
    }
    Some(DeviceSubmissionExecutionFacts::new(
        entries.len() as u32,
        commands,
    ))
}
