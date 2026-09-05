//! The wire format must affect actual token selection after protocol framing.
use super::{executor::LogitStep, *};

const PREFIX: &str = "{\"answer\":";
const INVALID_VALUE: &str = "\"wrong\"";
const VALID_VALUE: &str = "42";
const CLOSE: &str = "}";
const ENCODED_FINAL_HEADER: [&str; 7] = ["<|channel|>", "f", "i", "n", "a", "l", "<|message|>"];

async fn generate(protocol: ModelOutputProtocol, stream: bool, constrained: bool) -> Observation {
    let terminal = match protocol {
        ModelOutputProtocol::Text => ORDINARY_EOS,
        ModelOutputProtocol::HarmonyGptOss => RETURN,
        _ => panic!("unsupported structured protocol fixture"),
    };
    let mut pieces = Vec::new();
    if protocol == ModelOutputProtocol::HarmonyGptOss {
        pieces.extend(FINAL_HEADER);
    }
    pieces.extend([PREFIX, INVALID_VALUE, VALID_VALUE, CLOSE, terminal]);
    // Both candidates enter the ordinary BPE vocabulary. The unconstrained
    // control proves the invalid value is selectable by the real sampler.
    let tokenizer = Arc::new(tokenizer(&pieces).await);
    let token = |piece: &str| tokenizer.token_id(piece).unwrap();
    let mut steps = Vec::new();
    if protocol == ModelOutputProtocol::HarmonyGptOss {
        // The envelope grammar uses canonical tokenizer encoding for channel
        // names. This tiny BPE has no merges, so `final` spans five tokens.
        let header = tokenizer.encode(&FINAL_HEADER.concat(), false).unwrap();
        assert_eq!(
            header
                .iter()
                .map(|id| tokenizer.decode(&[*id], false).unwrap())
                .collect::<Vec<_>>(),
            ENCODED_FINAL_HEADER
        );
        steps.extend(header.into_iter().map(LogitStep::only));
    }
    steps.push(LogitStep::only(token(PREFIX)));
    steps.push(LogitStep::candidates(vec![
        (token(INVALID_VALUE), 10.0),
        (token(VALID_VALUE), 1.0),
    ]));
    steps.push(LogitStep::only(token(CLOSE)));
    steps.push(LogitStep::only(token(terminal)));
    let max_tokens = steps.len() + 4;
    let executor = Arc::new(ScriptedExecutor::from_steps(tokenizer.vocab_size(), steps));
    let format = if constrained {
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        })
    } else {
        json!({"type": "text"})
    };
    request_with_executor(
        protocol,
        tokenizer,
        executor,
        max_tokens,
        None,
        stream,
        Some(format),
    )
    .await
}

async fn assert_wire_schema_masks_only_the_payload(protocol: ModelOutputProtocol) {
    for stream in [false, true] {
        for constrained in [false, true] {
            let response = generate(protocol, stream, constrained).await;
            let value = if constrained {
                VALID_VALUE
            } else {
                INVALID_VALUE
            };
            let expected = format!("{PREFIX}{value}{CLOSE}");
            let mut expected_inputs = Vec::new();
            if protocol == ModelOutputProtocol::HarmonyGptOss {
                expected_inputs.extend(ENCODED_FINAL_HEADER.map(str::to_owned));
            }
            expected_inputs.extend([PREFIX, value, CLOSE].map(str::to_owned));
            // Observe the selected value as the next production decode input;
            // post-validating or truncating HTTP text cannot fake this choice.
            assert_eq!(
                response.decoded_inputs, expected_inputs,
                "{}",
                response.body
            );
            assert_success(&response, stream, &expected, expected_inputs.len() + 1);
        }
    }
}

#[tokio::test]
async fn wire_schema_changes_actual_text_sampling() {
    assert_wire_schema_masks_only_the_payload(ModelOutputProtocol::Text).await;
}

#[tokio::test]
async fn wire_schema_preserves_harmony_framing_before_constraining_payload() {
    assert_wire_schema_masks_only_the_payload(ModelOutputProtocol::HarmonyGptOss).await;
}
