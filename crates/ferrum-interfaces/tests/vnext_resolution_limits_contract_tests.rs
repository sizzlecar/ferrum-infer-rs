use ferrum_interfaces::vnext::*;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

fn artifact_id(value: &str) -> ResolutionArtifactId {
    ResolutionArtifactId::new(value).unwrap()
}

fn provenance() -> ResolutionSourceProvenance {
    ResolutionSourceProvenance::LockedModelFile {
        source_role: ModelArtifactSourceRole::Semantic,
        relative_path: "config.json".to_owned(),
    }
}

fn descriptor() -> Result<ResolutionParserDescriptor, VNextError> {
    ResolutionParserDescriptor::new(
        "resolution-parser.limits-test",
        ContractVersion::new(1, 0),
        ResolutionFingerprint::new("a".repeat(64))?,
    )
}

fn chosen_path() -> BTreeSet<String> {
    BTreeSet::from(["/chosen".to_owned()])
}

fn assert_invalid_field<T>(result: Result<T, VNextError>, expected_field: &str) {
    match result {
        Err(VNextError::InvalidResolvedModelPlan { field, .. }) => {
            assert_eq!(field, expected_field)
        }
        Err(error) => panic!("expected InvalidResolvedModelPlan, got {error}"),
        Ok(_) => panic!("expected `{expected_field}` rejection"),
    }
}

#[derive(Clone, Copy)]
enum DocumentShape {
    Small,
    Depth(usize),
    Nodes(usize),
    KeyAndStringBytes(usize),
}

struct ShapeParser {
    shape: DocumentShape,
    descriptor_calls: AtomicUsize,
    parse_calls: AtomicUsize,
}

impl ShapeParser {
    fn new(shape: DocumentShape) -> Self {
        Self {
            shape,
            descriptor_calls: AtomicUsize::new(0),
            parse_calls: AtomicUsize::new(0),
        }
    }

    fn reset_calls(&self) {
        self.descriptor_calls.store(0, Ordering::SeqCst);
        self.parse_calls.store(0, Ordering::SeqCst);
    }

    fn assert_calls(&self, descriptor_calls: usize, parse_calls: usize) {
        assert_eq!(
            self.descriptor_calls.load(Ordering::SeqCst),
            descriptor_calls
        );
        assert_eq!(self.parse_calls.load(Ordering::SeqCst), parse_calls);
    }

    fn document(&self) -> Value {
        match self.shape {
            DocumentShape::Small => json!({"chosen": null}),
            DocumentShape::Depth(depth) => {
                let mut value = Value::Null;
                for _ in 0..depth {
                    value = Value::Array(vec![value]);
                }
                value
            }
            DocumentShape::Nodes(nodes) => {
                assert!(nodes >= 2);
                json!({"chosen": vec![Value::Null; nodes - 2]})
            }
            DocumentShape::KeyAndStringBytes(bytes) => {
                const KEY_BYTES: usize = "chosen".len();
                assert!(bytes >= KEY_BYTES);
                json!({"chosen": "x".repeat(bytes - KEY_BYTES)})
            }
        }
    }
}

impl ResolutionSourceParser for ShapeParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        self.descriptor_calls.fetch_add(1, Ordering::SeqCst);
        descriptor()
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        _source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        self.parse_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.document())
    }
}

struct PathParser {
    keys: Vec<String>,
    descriptor_calls: AtomicUsize,
    parse_calls: AtomicUsize,
}

impl PathParser {
    fn new(paths: &BTreeSet<String>) -> Self {
        Self {
            keys: paths
                .iter()
                .map(|path| path.strip_prefix('/').unwrap().to_owned())
                .collect(),
            descriptor_calls: AtomicUsize::new(0),
            parse_calls: AtomicUsize::new(0),
        }
    }

    fn reset_calls(&self) {
        self.descriptor_calls.store(0, Ordering::SeqCst);
        self.parse_calls.store(0, Ordering::SeqCst);
    }

    fn assert_calls(&self, descriptor_calls: usize, parse_calls: usize) {
        assert_eq!(
            self.descriptor_calls.load(Ordering::SeqCst),
            descriptor_calls
        );
        assert_eq!(self.parse_calls.load(Ordering::SeqCst), parse_calls);
    }
}

impl ResolutionSourceParser for PathParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        self.descriptor_calls.fetch_add(1, Ordering::SeqCst);
        descriptor()
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        _source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        self.parse_calls.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Object(
            self.keys
                .iter()
                .map(|key| (key.clone(), Value::Null))
                .collect::<Map<_, _>>(),
        ))
    }
}

struct NondeterministicParser {
    descriptor_calls: AtomicUsize,
    parse_calls: AtomicUsize,
}

#[derive(Clone, Copy)]
enum InvalidDescriptorKind {
    EmptyId,
    ZeroMajor,
}

struct InvalidDescriptorParser {
    invalid_call: usize,
    kind: InvalidDescriptorKind,
    descriptor_calls: AtomicUsize,
    parse_calls: AtomicUsize,
}

impl InvalidDescriptorParser {
    fn new(invalid_call: usize, kind: InvalidDescriptorKind) -> Self {
        Self {
            invalid_call,
            kind,
            descriptor_calls: AtomicUsize::new(0),
            parse_calls: AtomicUsize::new(0),
        }
    }

    fn invalid_descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        let (id, major) = match self.kind {
            InvalidDescriptorKind::EmptyId => ("", 1),
            InvalidDescriptorKind::ZeroMajor => ("resolution-parser.invalid", 0),
        };
        serde_json::from_value(json!({
            "id": id,
            "version": {"major": major, "minor": 0},
            "implementation_fingerprint": "a".repeat(64),
        }))
        .map_err(|error| VNextError::Serialization {
            context: "deserialize adversarial resolution parser descriptor",
            message: error.to_string(),
        })
    }
}

impl ResolutionSourceParser for InvalidDescriptorParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        let call = self.descriptor_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.invalid_call {
            self.invalid_descriptor()
        } else {
            descriptor()
        }
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        _source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        self.parse_calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"chosen": null}))
    }
}

impl ResolutionSourceParser for NondeterministicParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        self.descriptor_calls.fetch_add(1, Ordering::SeqCst);
        descriptor()
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        _source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        let call = self.parse_calls.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"chosen": call}))
    }
}

fn fixed_length_paths(count: usize, length: usize) -> BTreeSet<String> {
    (0..count)
        .map(|index| {
            let prefix = format!("/{index:04x}");
            assert!(prefix.len() <= length);
            format!("{prefix}{}", "p".repeat(length - prefix.len()))
        })
        .collect()
}

#[test]
fn public_resolution_availability_limits_are_stable() {
    assert_eq!(MAX_RESOLUTION_SOURCE_BYTES, 32 * 1024 * 1024);
    assert_eq!(MAX_RESOLVED_MODEL_PLAN_WIRE_BYTES, 16 * 1024 * 1024);
    assert_eq!(MAX_RESOLUTION_JSON_DEPTH, 128);
    assert_eq!(MAX_RESOLUTION_JSON_NODES, 1_000_000);
    assert_eq!(
        MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES,
        MAX_RESOLUTION_SOURCE_BYTES
    );
    assert_eq!(MAX_RESOLUTION_PROVENANCE_BYTES, 4 * 1024);
    assert_eq!(MAX_RESOLUTION_FIELD_PATHS, 4_096);
    assert_eq!(MAX_RESOLUTION_FIELD_PATH_BYTES, 512);
    assert_eq!(MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES, 1024 * 1024);
}

#[test]
fn source_byte_limit_accepts_max_and_rejects_max_plus_one_before_parser() {
    let parser = ShapeParser::new(DocumentShape::Small);
    let evidence = ResolutionSourceEvidence::new(
        artifact_id("artifact.source-max"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        vec![b'x'; MAX_RESOLUTION_SOURCE_BYTES],
        chosen_path(),
        &parser,
    )
    .unwrap();
    parser.assert_calls(0, 0);
    evidence.validate().unwrap();
    parser.assert_calls(3, 2);
    drop(evidence);

    parser.reset_calls();
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.source-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            vec![b'x'; MAX_RESOLUTION_SOURCE_BYTES + 1],
            chosen_path(),
            &parser,
        ),
        "resolution_source_evidence.source_bytes",
    );
    parser.assert_calls(0, 0);
}

#[test]
fn json_parser_checks_source_bytes_when_called_directly() {
    let provenance = provenance();
    let mut bytes = vec![b' '; MAX_RESOLUTION_SOURCE_BYTES];
    bytes[..4].copy_from_slice(b"null");
    assert_eq!(
        JSON_RESOLUTION_SOURCE_PARSER
            .parse(ResolutionDecisionSource::ModelMetadata, &provenance, &bytes,)
            .unwrap(),
        Value::Null
    );

    bytes.push(b' ');
    assert_invalid_field(
        JSON_RESOLUTION_SOURCE_PARSER.parse(
            ResolutionDecisionSource::ModelMetadata,
            &provenance,
            &bytes,
        ),
        "resolution_source_evidence.source_bytes",
    );
}

fn flat_json_array(nodes: usize) -> Vec<u8> {
    assert!(nodes >= 1);
    let children = nodes - 1;
    let mut bytes = Vec::with_capacity(children.saturating_mul(2).saturating_add(1));
    bytes.push(b'[');
    for index in 0..children {
        if index != 0 {
            bytes.push(b',');
        }
        bytes.push(b'0');
    }
    bytes.push(b']');
    bytes
}

#[test]
fn json_parser_preflight_enforces_node_budget_before_building_a_value_tree() {
    let provenance = provenance();
    let at_limit = flat_json_array(MAX_RESOLUTION_JSON_NODES);
    assert!(at_limit.len() < MAX_RESOLUTION_SOURCE_BYTES);
    let document = JSON_RESOLUTION_SOURCE_PARSER
        .parse(
            ResolutionDecisionSource::ModelMetadata,
            &provenance,
            &at_limit,
        )
        .unwrap();
    assert_eq!(
        document.as_array().unwrap().len() + 1,
        MAX_RESOLUTION_JSON_NODES
    );
    drop(document);

    let over_limit = flat_json_array(MAX_RESOLUTION_JSON_NODES + 1);
    assert!(over_limit.len() < MAX_RESOLUTION_SOURCE_BYTES);
    assert_invalid_field(
        JSON_RESOLUTION_SOURCE_PARSER.parse(
            ResolutionDecisionSource::ModelMetadata,
            &provenance,
            &over_limit,
        ),
        "resolution_source_evidence.document.nodes",
    );
}

#[test]
fn json_parser_preflight_handles_escaped_structure_and_rejects_trailing_roots() {
    let provenance = provenance();
    let bytes = br#"{"key\u005b":"brackets [ ] braces { } comma , quote \" slash \\","values":[-1,2.5e3,true,false,null]}"#;
    assert_eq!(
        JSON_RESOLUTION_SOURCE_PARSER
            .parse(ResolutionDecisionSource::ModelMetadata, &provenance, bytes)
            .unwrap(),
        json!({
            "key[": "brackets [ ] braces { } comma , quote \" slash \\",
            "values": [-1, 2.5e3, true, false, null]
        })
    );

    match JSON_RESOLUTION_SOURCE_PARSER.parse(
        ResolutionDecisionSource::ModelMetadata,
        &provenance,
        b"null []",
    ) {
        Err(VNextError::Serialization { context, .. }) => {
            assert_eq!(context, "parse resolution source JSON")
        }
        Err(error) => panic!("trailing root hit the wrong gate: {error}"),
        Ok(_) => panic!("multiple JSON roots must be rejected"),
    }
}

#[test]
fn json_parser_preflight_reports_depth_budget_before_serde_recursion_failure() {
    let provenance = provenance();
    let mut bytes = vec![b'['; MAX_RESOLUTION_JSON_DEPTH + 1];
    bytes.push(b'0');
    bytes.extend(std::iter::repeat_n(b']', MAX_RESOLUTION_JSON_DEPTH + 1));
    assert_invalid_field(
        JSON_RESOLUTION_SOURCE_PARSER.parse(
            ResolutionDecisionSource::ModelMetadata,
            &provenance,
            &bytes,
        ),
        "resolution_source_evidence.document.depth",
    );
}

#[test]
fn provenance_limit_accepts_max_and_rejects_max_plus_one_before_parser() {
    let parser = ShapeParser::new(DocumentShape::Small);
    let maximum_path_bytes =
        MAX_RESOLUTION_PROVENANCE_BYTES - ModelArtifactSourceRole::Semantic.as_str().len();
    let evidence = ResolutionSourceEvidence::new(
        artifact_id("artifact.provenance-max"),
        ResolutionDecisionSource::ModelMetadata,
        ResolutionSourceProvenance::LockedModelFile {
            source_role: ModelArtifactSourceRole::Semantic,
            relative_path: "p".repeat(maximum_path_bytes),
        },
        b"ignored".to_vec(),
        chosen_path(),
        &parser,
    )
    .unwrap();
    parser.assert_calls(0, 0);
    evidence.validate().unwrap();
    parser.assert_calls(3, 2);
    drop(evidence);

    parser.reset_calls();
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.provenance-over"),
            ResolutionDecisionSource::ModelMetadata,
            ResolutionSourceProvenance::LockedModelFile {
                source_role: ModelArtifactSourceRole::Semantic,
                relative_path: "p".repeat(maximum_path_bytes + 1),
            },
            b"ignored".to_vec(),
            chosen_path(),
            &parser,
        ),
        "resolution_source_evidence.provenance",
    );
    parser.assert_calls(0, 0);
}

#[test]
fn field_path_count_and_total_bytes_are_bounded_before_parser() {
    assert_eq!(
        MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES % MAX_RESOLUTION_FIELD_PATHS,
        0
    );
    let path_length = MAX_RESOLUTION_FIELD_PATH_TOTAL_BYTES / MAX_RESOLUTION_FIELD_PATHS;
    let exact_paths = fixed_length_paths(MAX_RESOLUTION_FIELD_PATHS, path_length);
    let parser = PathParser::new(&exact_paths);
    let evidence = ResolutionSourceEvidence::new(
        artifact_id("artifact.paths-max"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        exact_paths.clone(),
        &parser,
    )
    .unwrap();
    parser.assert_calls(0, 0);
    evidence.validate().unwrap();
    parser.assert_calls(3, 2);
    drop(evidence);

    let mut over_total = exact_paths;
    let first = over_total.pop_first().unwrap();
    over_total.insert(format!("{first}x"));
    parser.reset_calls();
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.paths-total-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            over_total,
            &parser,
        ),
        "resolution_source_evidence.field_paths",
    );
    parser.assert_calls(0, 0);

    let over_count = (0..=MAX_RESOLUTION_FIELD_PATHS)
        .map(|index| format!("/p{index}"))
        .collect();
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.paths-count-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            over_count,
            &parser,
        ),
        "resolution_source_evidence.field_paths",
    );
    parser.assert_calls(0, 0);
}

#[test]
fn parsed_json_depth_node_and_text_budgets_are_enforced_for_each_result() {
    let depth_at_limit = ShapeParser::new(DocumentShape::Depth(MAX_RESOLUTION_JSON_DEPTH));
    ResolutionSourceEvidence::new(
        artifact_id("artifact.depth-max"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        BTreeSet::from(["/0".to_owned()]),
        &depth_at_limit,
    )
    .unwrap()
    .validate()
    .unwrap();
    depth_at_limit.assert_calls(3, 2);

    let depth_over = ShapeParser::new(DocumentShape::Depth(MAX_RESOLUTION_JSON_DEPTH + 1));
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.depth-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            BTreeSet::from(["/0".to_owned()]),
            &depth_over,
        )
        .and_then(|evidence| evidence.validate()),
        "resolution_source_evidence.document.depth",
    );
    depth_over.assert_calls(1, 1);

    let nodes_at_limit = ShapeParser::new(DocumentShape::Nodes(MAX_RESOLUTION_JSON_NODES));
    ResolutionSourceEvidence::new(
        artifact_id("artifact.nodes-max"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        chosen_path(),
        &nodes_at_limit,
    )
    .unwrap()
    .validate()
    .unwrap();
    nodes_at_limit.assert_calls(3, 2);

    let nodes_over = ShapeParser::new(DocumentShape::Nodes(MAX_RESOLUTION_JSON_NODES + 1));
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.nodes-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            chosen_path(),
            &nodes_over,
        )
        .and_then(|evidence| evidence.validate()),
        "resolution_source_evidence.document.nodes",
    );
    nodes_over.assert_calls(1, 1);

    let text_at_limit = ShapeParser::new(DocumentShape::KeyAndStringBytes(
        MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES,
    ));
    ResolutionSourceEvidence::new(
        artifact_id("artifact.text-max"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        chosen_path(),
        &text_at_limit,
    )
    .unwrap()
    .validate()
    .unwrap();
    text_at_limit.assert_calls(3, 2);

    let text_over = ShapeParser::new(DocumentShape::KeyAndStringBytes(
        MAX_RESOLUTION_JSON_KEY_AND_STRING_BYTES + 1,
    ));
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.text-over"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            chosen_path(),
            &text_over,
        )
        .and_then(|evidence| evidence.validate()),
        "resolution_source_evidence.document.key_and_string_bytes",
    );
    text_over.assert_calls(1, 1);
}

#[test]
fn repeated_parser_results_must_be_canonically_deterministic() {
    let parser = NondeterministicParser {
        descriptor_calls: AtomicUsize::new(0),
        parse_calls: AtomicUsize::new(0),
    };
    assert_invalid_field(
        ResolutionSourceEvidence::new(
            artifact_id("artifact.nondeterministic"),
            ResolutionDecisionSource::ProductDefault,
            provenance(),
            b"ignored".to_vec(),
            chosen_path(),
            &parser,
        )
        .and_then(|evidence| evidence.validate()),
        "resolution_source_evidence.parser",
    );
    assert_eq!(parser.descriptor_calls.load(Ordering::SeqCst), 3);
    assert_eq!(parser.parse_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn parser_descriptor_deserialization_and_each_verification_read_fail_closed() {
    for invalid in [
        json!({
            "id": "",
            "version": {"major": 1, "minor": 0},
            "implementation_fingerprint": "a".repeat(64),
        }),
        json!({
            "id": "resolution-parser.invalid",
            "version": {"major": 0, "minor": 0},
            "implementation_fingerprint": "a".repeat(64),
        }),
    ] {
        assert!(serde_json::from_value::<ResolutionParserDescriptor>(invalid).is_err());
    }

    let invalid_second = InvalidDescriptorParser::new(2, InvalidDescriptorKind::EmptyId);
    let evidence = ResolutionSourceEvidence::new(
        artifact_id("artifact.invalid-second-descriptor"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        chosen_path(),
        &invalid_second,
    )
    .unwrap();
    assert!(evidence.validate().is_err());
    assert_eq!(invalid_second.descriptor_calls.load(Ordering::SeqCst), 2);
    assert_eq!(invalid_second.parse_calls.load(Ordering::SeqCst), 1);

    let invalid_third = InvalidDescriptorParser::new(3, InvalidDescriptorKind::ZeroMajor);
    let evidence = ResolutionSourceEvidence::new(
        artifact_id("artifact.invalid-third-descriptor"),
        ResolutionDecisionSource::ProductDefault,
        provenance(),
        b"ignored".to_vec(),
        chosen_path(),
        &invalid_third,
    )
    .unwrap();
    assert!(evidence.validate().is_err());
    assert_eq!(invalid_third.descriptor_calls.load(Ordering::SeqCst), 3);
    assert_eq!(invalid_third.parse_calls.load(Ordering::SeqCst), 2);
}

#[test]
fn resolved_wire_limit_accepts_max_and_rejects_max_plus_one_before_serde() {
    let mut bytes = vec![b' '; MAX_RESOLVED_MODEL_PLAN_WIRE_BYTES];
    match ResolvedModelPlan::decode_untrusted(&bytes) {
        Err(VNextError::Serialization { context, .. }) => {
            assert_eq!(context, "decode untrusted resolved model plan")
        }
        Err(error) => panic!("equal-to-max wire bytes hit the wrong gate: {error}"),
        Ok(_) => panic!("whitespace is not a resolved plan"),
    }

    bytes.push(b' ');
    assert_invalid_field(
        ResolvedModelPlan::decode_untrusted(&bytes),
        "resolved_model_plan.wire_bytes",
    );
}
