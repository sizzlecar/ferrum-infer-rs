//! Source6 ordinary service windows. Open frontiers are explicit and no
//! complete-request/Length claim is made. Profiles still require strict replay.
use super::*;
mod owner_blocks;
pub use owner_blocks::*;
mod no_submission;
pub use no_submission::StructuredServiceNoSubmissionV6;
mod route_population;
pub(super) use route_population::OutsideEvidence;
pub use route_population::{StructuredServiceOutsideRouteV6, StructuredServiceRouteCountsV1};
mod owner_diagnostic;
mod physical;
pub use owner_diagnostic::{
    diagnose_structured_service_owners_v6, StructuredOutsideOwnerDiagnosticV1,
    StructuredOwnerDifferenceV1, StructuredOwnerReplayFailureV1,
    StructuredServiceOwnerDiagnosticV1,
};
mod replay;
mod wire;
pub use replay::StructuredServiceCollectorV6;
mod profile;
pub use profile::{
    export_structured_profile_v13, load_structured_profile_v13, ImportedStructuredCatalogV13,
    StructuredProfileExportReceiptV13,
};
use wire::*;
#[cfg(test)]
mod tests;
pub use wire::{
    StructuredServiceChildFreezeV6, StructuredServiceClockV6, StructuredServiceDeclarationV6,
    StructuredServiceDomainFreezeV1, StructuredServiceHeaderV6, StructuredServiceRecordV6,
    StructuredServiceWaveV6,
};
pub const SERVICE_SOURCE_PROTOCOL_V6: &str = "ferrum.structured-service-windows.v1";
fn phase_index(phase: StructuredPhaseV2) -> usize {
    match phase {
        StructuredPhaseV2::Fit => 0,
        StructuredPhaseV2::Residual => 1,
        StructuredPhaseV2::Qualification => 2,
    }
}
fn phase_at(i: usize) -> StructuredPhaseV2 {
    match i {
        0 => StructuredPhaseV2::Fit,
        1 => StructuredPhaseV2::Residual,
        _ => StructuredPhaseV2::Qualification,
    }
}
