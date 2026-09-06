//! Distribution checks bind existing CI and execution, not model correctness.
use super::{Behavior, CheckDescriptor, EvidenceLayer};

pub fn distribution_check_descriptors() -> Vec<CheckDescriptor> {
    vec![
        CheckDescriptor {
            id: "release-delivery.workspace".into(),
            behavior: Behavior::WorkspaceChecks,
            layer: EvidenceLayer::Compilation,
            entrypoints: Vec::new(),
            target: None,
        },
        CheckDescriptor {
            id: "release-delivery.installation".into(),
            behavior: Behavior::Installation,
            layer: EvidenceLayer::Installation,
            entrypoints: Vec::new(),
            target: None,
        },
    ]
}
