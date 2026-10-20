//! Bounded JSONL publication with one writer and explicit lifecycle ownership.
//!
//! Producers borrow publication authority from their component scope. Only the
//! original owner may finally close and join the writer. Event schemas and the
//! caller's required-evidence policy remain with their producing components.

pub mod jsonl_journal;

pub use jsonl_journal::{
    normalize_jsonl_path, write_jsonl_records, write_jsonl_records_to, JsonlFinalReceipt,
    JsonlFlushReceipt, JsonlJournal, JsonlJournalConfig, JsonlJournalDeliveryStats,
    JsonlJournalError, JsonlJournalErrorKind, JsonlJournalOpenMode, JsonlJournalOwner,
    JsonlJournalWorkerState, JsonlProducerScope, JsonlPublicationCut, OwnedJsonlJournal,
};
