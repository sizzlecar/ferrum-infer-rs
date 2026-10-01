//! Original global offers are recorded once; each declared owner progresses
//! independently across complete blocks. This protocol does not reinterpret V6.
use super::*;
mod prepared;
pub use prepared::*;
mod canonical;
pub use canonical::{canonical_value_v7, record_bytes_v7};
mod wire;
pub use wire::*;
mod no_submission;
pub use no_submission::StructuredServiceNoSubmissionV7;
mod collector;
mod discovery;
pub use collector::{
    replay_structured_source_v7, StructuredServiceCollectorV7, StructuredSourceRecordSinkV1,
};
mod profile;
pub use profile::{
    export_structured_profile_v14, export_structured_profile_v14_same_boot,
    export_structured_profile_v14_same_boot_selected,
    export_structured_profile_v14_same_boot_selected_from_original_bytes,
    load_structured_profile_v14, load_structured_profile_v14_same_boot,
    load_structured_profile_v14_same_boot_selected,
    load_structured_profile_v14_same_boot_selected_from_original_bytes,
    structured_owner_block_metadata_maximum_bytes, ImportedStructuredCatalogV14,
    StructuredProfileExportReceiptV14, StructuredServiceCheckpointV7,
};
pub const SERVICE_SOURCE_PROTOCOL_V7: &str = "ferrum.structured-owner-blocks.v1";
