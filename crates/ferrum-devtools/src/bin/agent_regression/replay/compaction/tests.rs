use super::*;
use std::io::Write;

struct Fixture {
    entries: Vec<Value>,
    origin: RequestRecord,
    following: RequestRecord,
    events: Events,
}

fn assistant(id: &str, parent: &str, response: &str) -> Value {
    json!({"type":"message", "id":id, "parentId":parent, "message":{
    "role":"assistant", "responseId":response, "content":[
        {"type":"text", "text":"Read source.\n"},
        {"type":"toolCall", "id":"call_0", "name":"read", "arguments":{"path":"src/lib.rs"}}
    ]}})
}

fn result(id: &str, parent: &str) -> Value {
    json!({"type":"message", "id":id, "parentId":parent, "message":{
        "role":"toolResult", "toolCallId":"call_0", "toolName":"read", "isError":false,
        "content":[{"type":"text", "text":" source\n"}]}})
}

fn assistant_wire() -> Value {
    json!({"role":"assistant", "content":"Read source.\n", "tool_calls":[{
        "id":"call_0", "type":"function", "function":{
            "name":"read", "arguments":"{\"path\":\"src/lib.rs\"}"}}]})
}

fn result_wire() -> Value {
    json!({"role":"tool", "tool_call_id":"call_0", "content":" source\n"})
}

impl Fixture {
    fn new() -> Self {
        let entries = vec![
            json!({"type":"session", "version":3, "id":"observed-session"}),
            json!({"type":"message", "id":"user", "parentId":null, "message":{
                "role":"user", "content":[{"type":"text", "text":"Fix the original task"}]}}),
            assistant("old-assistant", "user", "old-response"),
            result("old-result", "old-assistant"),
            assistant("origin-assistant", "old-result", "origin-response"),
            result("origin-result", "origin-assistant"),
            json!({"type":"compaction", "id":"compacted", "parentId":"origin-result",
                "firstKeptEntryId":"old-assistant", "summary":"Actual summary"}),
            assistant("following-assistant", "compacted", "following-response"),
        ];
        let system = json!({"role":"system", "content":"Local coding assistant"});
        let origin = RequestRecord {
            request_index: 4,
            server_request_id: Some("origin-response".into()),
            messages: vec![
                system.clone(),
                json!({"role":"user", "content":[{"type":"text", "text":"Fix the original task"}]}),
                assistant_wire(),
                result_wire(),
            ],
            http_status: Some(200),
            saw_done: true,
            finish_reasons: vec!["tool_calls".into()],
            ..Default::default()
        };
        let following = RequestRecord {
            request_index: 6,
            server_request_id: Some("following-response".into()),
            messages: vec![
                system,
                json!({"role":"user", "content":[{"type":"text",
                "text":format!("{SUMMARY_PREFIX}Actual summary{SUMMARY_SUFFIX}")}]}),
                assistant_wire(),
                result_wire(),
                assistant_wire(),
                result_wire(),
            ],
            http_status: Some(200),
            saw_done: true,
            finish_reasons: vec!["stop".into()],
            ..Default::default()
        };
        let mut events = Events::default();
        events.session_id = Some("observed-session".into());
        events.tools.insert(
            "2:call_0".into(),
            Tool {
                call_id: "call_0".into(),
                assistant_turn: 2,
                origin_response_id: Some("origin-response".into()),
                arguments: json!({"path":"src/lib.rs"}),
                name: "read".into(),
                requested: true,
                started_ns: Some(10),
                ended_ns: Some(20),
                returned_ns: Some(30),
                returned_to_session: true,
                result_text: Some(" source\n".into()),
                ..Default::default()
            },
        );
        Self {
            entries,
            origin,
            following,
            events,
        }
    }

    fn verify(&self) -> Evidence {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        for entry in &self.entries {
            writeln!(file, "{entry}").unwrap();
        }
        verify_with_session(&self.events, &[&self.origin, &self.following], file.path())
    }

    fn rejects(&self, reason: &str) {
        let evidence = self.verify();
        assert!(!evidence.complete(&self.events));
        assert!(evidence.verified.is_empty());
        assert!(
            evidence.unproven["2:call_0"].contains(reason),
            "{:?}",
            evidence.unproven
        );
    }
}

#[test]
fn retained_branch_binds_exact_origin_and_result_after_real_history_replacement() {
    let fixture = Fixture::new();
    assert!(
        !super::super::verify(&fixture.events, &[&fixture.origin, &fixture.following])
            .complete(&fixture.events)
    );
    let evidence = fixture.verify();
    assert!(evidence.complete(&fixture.events));
    let receipt = &evidence.verified[0];
    assert_eq!(receipt.origin_request_index, 4);
    assert_eq!(receipt.following_request_index, 6);
    assert_eq!(
        (receipt.assistant_message_index, receipt.tool_message_index),
        (4, 5)
    );
    let compaction = receipt.compaction.as_ref().unwrap();
    assert_eq!(compaction.origin_assistant_entry_id, "origin-assistant");
    assert_eq!(compaction.result_entry_id, "origin-result");
    assert_eq!(compaction.first_kept_entry_id, "old-assistant");
    assert!(!compaction.session_sha256.is_empty());
}

#[test]
fn identical_old_call_and_result_cannot_witness_a_different_branch() {
    let mut fixture = Fixture::new();
    fixture.entries[6]["parentId"] = "old-result".into();
    fixture.following.messages.truncate(4);
    fixture.rejects("origin assistant is not a compaction ancestor");
    let mut fixture = Fixture::new();
    fixture.entries[7]["parentId"] = "old-result".into();
    fixture.rejects("no successful request contains the complete bound compaction prefix");
}

#[test]
fn retained_origin_requires_source_response_id_and_valid_first_kept_ancestor() {
    let mut fixture = Fixture::new();
    fixture.entries[4]["message"]["responseId"] = "different-response".into();
    fixture.rejects("no unique session assistant");
    let mut fixture = Fixture::new();
    fixture.entries[6]["firstKeptEntryId"] = "absent".into();
    fixture.rejects("first kept entry is not an ancestor");
    let mut fixture = Fixture::new();
    fixture.entries[6]["firstKeptEntryId"] = "origin-result".into();
    fixture.rejects("did not retain the origin assistant");
}

#[test]
fn exact_result_and_entire_compacted_prefix_are_required() {
    for mutate in [0, 1, 2, 3, 4] {
        let mut fixture = Fixture::new();
        match mutate {
            0 => fixture.following.messages[5]["content"] = "source\n".into(),
            1 => fixture.following.messages[1]["content"][0]["text"] = "A different summary".into(),
            2 => {
                fixture.following.messages[4]["tool_calls"][0]["function"]["arguments"] =
                    "{\"path\":\"other\"}".into()
            }
            3 => fixture.following.messages[0]["content"] = "Different system".into(),
            _ => fixture.following.error = Some("response failed".into()),
        }
        fixture.rejects("no successful request contains the complete bound compaction prefix");
    }
    let mut fixture = Fixture::new();
    fixture.entries[5]["message"]["content"][0]["text"] = "source\n".into();
    fixture.rejects("session tool result differs");
}

#[test]
fn session_and_wire_must_agree_about_retained_pre_origin_context() {
    let mut fixture = Fixture::new();
    fixture.origin.messages[3]["content"] = "different previous result".into();
    fixture.rejects("retained pre-origin entries differ");
    let mut fixture = Fixture::new();
    fixture.entries[2]["type"] = "custom_message".into();
    fixture.rejects("unsupported retained context entry");
}

#[test]
fn following_response_must_include_all_actual_post_compaction_context() {
    let mut fixture = Fixture::new();
    fixture.entries.insert(
        7,
        json!({"type":"message", "id":"feedback", "parentId":"compacted", "message":{
        "role":"user", "content":[{"type":"text", "text":"Continue the same task"}]}}),
    );
    fixture.entries[8]["parentId"] = "feedback".into();
    fixture.rejects("no successful request contains the complete bound compaction prefix");
    fixture
        .following
        .messages
        .push(json!({"role":"user", "content":[{"type":"text", "text":"Continue the same task"}]}));
    assert!(fixture.verify().complete(&fixture.events));
}

#[test]
fn missing_session_does_not_revoke_ordinary_exact_replay() {
    let mut fixture = Fixture::new();
    fixture.following.messages = fixture.origin.messages.clone();
    fixture
        .following
        .messages
        .extend([assistant_wire(), result_wire()]);
    let dir = tempfile::tempdir().unwrap();
    let evidence = verify_with_session(
        &fixture.events,
        &[&fixture.origin, &fixture.following],
        &dir.path().join("missing"),
    );
    assert!(evidence.complete(&fixture.events));
    assert!(evidence.verified[0].compaction.is_none());
}

#[test]
fn missing_or_malformed_session_keeps_unproven_instead_of_losing_the_report() {
    let mut fixture = Fixture::new();
    fixture.entries[0]["id"] = "other-session".into();
    fixture.rejects("session header/version differs");
    let mut fixture = Fixture::new();
    fixture.entries.push(fixture.entries[5].clone());
    fixture.rejects("duplicate session entry id");
    let fixture = Fixture::new();
    let empty_dir = tempfile::tempdir().unwrap();
    let evidence = verify_with_session(
        &fixture.events,
        &[&fixture.origin, &fixture.following],
        &empty_dir.path().join("missing"),
    );
    assert!(!evidence.complete(&fixture.events));
    assert!(evidence.unproven["2:call_0"].contains("session evidence unavailable"));
}
