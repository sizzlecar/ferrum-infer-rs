//! Shared resource lifecycle event envelope for offline invariant checks.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceAction {
    RequestOpen,
    Reserve,
    Commit,
    Defer,
    Reject,
    Release,
    Rollback,
    RequestClose,
    CapacitySnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTraceEvent {
    pub owner_kind: String,
    pub owner_id: String,
    pub resource_kind: String,
    pub action: ResourceAction,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capacity: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underflow_amount: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_error_kind: Option<String>,
}

impl ResourceTraceEvent {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.owner_kind.trim().is_empty() {
            return Err("resource owner_kind must be non-empty".to_string());
        }
        if self.owner_id.trim().is_empty() {
            return Err("resource owner_id must be non-empty".to_string());
        }
        if self.resource_kind.trim().is_empty() {
            return Err("resource_kind must be non-empty".to_string());
        }
        if self.error_kind.as_deref().is_some_and(str::is_empty) {
            return Err("resource error_kind must be non-empty when set".to_string());
        }
        if self.message.as_deref().is_some_and(str::is_empty) {
            return Err("resource message must be non-empty when set".to_string());
        }
        if self
            .resource_error_kind
            .as_deref()
            .is_some_and(str::is_empty)
        {
            return Err("resource resource_error_kind must be non-empty when set".to_string());
        }
        if self.underflow_amount.is_some_and(|amount| amount <= 0) {
            return Err("resource underflow_amount must be positive when set".to_string());
        }
        match self.action {
            ResourceAction::Reserve
            | ResourceAction::Commit
            | ResourceAction::Release
            | ResourceAction::Rollback => {
                if self.amount.is_none() {
                    return Err("resource amount is required for lifecycle action".to_string());
                }
                if self.before.is_none() || self.after.is_none() {
                    return Err(
                        "resource before/after are required for lifecycle action".to_string()
                    );
                }
            }
            ResourceAction::Defer | ResourceAction::Reject => {
                if self.reason.as_deref().unwrap_or("").trim().is_empty() {
                    return Err("resource defer/reject reason must be non-empty".to_string());
                }
            }
            ResourceAction::CapacitySnapshot => {
                if self.capacity.is_none() {
                    return Err("resource capacity is required for capacity_snapshot".to_string());
                }
            }
            ResourceAction::RequestOpen | ResourceAction::RequestClose => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_event_requires_owner_and_before_after() {
        let event = ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: "req-1".to_string(),
            resource_kind: "kv_block".to_string(),
            action: ResourceAction::Reserve,
            amount: Some(1),
            before: Some(4),
            after: Some(3),
            capacity: Some(4),
            underflow_amount: None,
            reason: None,
            error_kind: None,
            message: None,
            resource_error_kind: None,
        };
        event.validate().unwrap();

        let mut missing_owner = event.clone();
        missing_owner.owner_id.clear();
        assert!(missing_owner.validate().is_err());

        let mut missing_after = event;
        missing_after.after = None;
        assert!(missing_after.validate().is_err());
    }

    #[test]
    fn lifecycle_event_allows_failure_diagnostics_when_non_empty() {
        let mut event = ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: "req-oom".to_string(),
            resource_kind: "backend_workspace".to_string(),
            action: ResourceAction::RequestClose,
            amount: None,
            before: None,
            after: None,
            capacity: None,
            underflow_amount: None,
            reason: None,
            error_kind: Some("cuda_oom".to_string()),
            message: Some("CUDA out of memory".to_string()),
            resource_error_kind: Some("kv_capacity_exhausted".to_string()),
        };
        event.validate().unwrap();

        event.error_kind = Some(String::new());
        assert!(event.validate().is_err());
    }

    #[test]
    fn lifecycle_event_allows_positive_underflow_amount_only() {
        let mut event = ResourceTraceEvent {
            owner_kind: "request".to_string(),
            owner_id: "req-underflow".to_string(),
            resource_kind: "kv_block".to_string(),
            action: ResourceAction::Release,
            amount: Some(2),
            before: Some(1),
            after: Some(-1),
            capacity: Some(4),
            underflow_amount: Some(1),
            reason: None,
            error_kind: None,
            message: None,
            resource_error_kind: None,
        };
        event.validate().unwrap();

        event.underflow_amount = Some(0);
        assert!(event.validate().is_err());
    }
}
