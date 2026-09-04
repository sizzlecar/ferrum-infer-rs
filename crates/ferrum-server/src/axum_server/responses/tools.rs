use super::super::ServerError;
use super::request::{
    optional_bool, optional_string, reject_unknown_object_fields, required_string, unsupported,
};
use super::{ResponseToolName, ToolNameMap};
use crate::openai::{ChatFunction, ChatTool, ToolChoice, ToolChoiceFunction};
use serde_json::{json, Map, Value};
use std::collections::HashSet;

pub(super) const MAX_CHAT_TOOL_NAME_BYTES: usize = 64;

pub(super) fn parse_tools(
    tools: Option<&[Value]>,
) -> std::result::Result<(Vec<ChatTool>, Vec<Value>, ToolNameMap), ServerError> {
    let tools = tools.unwrap_or_default();
    let mut chat_tools = Vec::new();
    let mut response_tools = Vec::new();
    let mut tool_names = ToolNameMap::default();
    let mut response_names = HashSet::new();
    let mut occupied_chat_names = HashSet::new();

    // Reserve every plain function name before allocating namespace aliases so
    // the result cannot depend on whether a namespace appears before or after
    // a colliding plain function in the request.
    for (index, tool) in tools.iter().enumerate() {
        let object = tool.as_object().ok_or_else(|| {
            ServerError::invalid_request(
                "tools entries must be objects",
                Some(&format!("tools[{index}]")),
            )
        })?;
        let type_param = format!("tools[{index}].type");
        let tool_type = required_string(object, "type", &type_param)?;
        if tool_type == "function" {
            let name = required_string(object, "name", &format!("tools[{index}].name"))?;
            occupied_chat_names.insert(name);
        }
    }

    for (index, tool) in tools.iter().enumerate() {
        let object = tool.as_object().ok_or_else(|| {
            ServerError::invalid_request(
                "tools entries must be objects",
                Some(&format!("tools[{index}]")),
            )
        })?;
        let parent = format!("tools[{index}]");
        let type_param = format!("{parent}.type");
        let tool_type = required_string(object, "type", &type_param)?;
        match tool_type.as_str() {
            "function" => {
                let (function, normalized) = parse_function_tool(object, &parent)?;
                let response_name = ResponseToolName {
                    namespace: None,
                    name: function.name.clone(),
                };
                if !response_names.insert(response_name.clone()) {
                    return Err(ServerError::invalid_request(
                        format!("duplicate function tool `{}`", function.name),
                        Some(&format!("{parent}.name")),
                    ));
                }
                tool_names.insert(response_name, function.name.clone());
                chat_tools.push(ChatTool {
                    tool_type: "function".to_string(),
                    function,
                });
                response_tools.push(normalized);
            }
            "namespace" => {
                reject_unknown_object_fields(
                    object,
                    &["type", "name", "description", "tools"],
                    &parent,
                )?;
                let namespace = required_string(object, "name", &format!("{parent}.name"))?;
                let namespace_description =
                    optional_string(object, "description", &format!("{parent}.description"))?;
                let nested = object
                    .get("tools")
                    .and_then(Value::as_array)
                    .ok_or_else(|| {
                        ServerError::invalid_request(
                            "namespace tools must be an array",
                            Some(&format!("{parent}.tools")),
                        )
                    })?;
                let mut normalized_nested = Vec::with_capacity(nested.len());
                for (nested_index, nested_tool) in nested.iter().enumerate() {
                    let nested_parent = format!("{parent}.tools[{nested_index}]");
                    let nested_object = nested_tool.as_object().ok_or_else(|| {
                        ServerError::invalid_request(
                            "namespace tool entries must be objects",
                            Some(&nested_parent),
                        )
                    })?;
                    let nested_type =
                        required_string(nested_object, "type", &format!("{nested_parent}.type"))?;
                    if nested_type != "function" {
                        return Err(unsupported(
                            format!(
                                "namespace tool type `{nested_type}` is not supported; only nested function tools are supported"
                            ),
                            &format!("{nested_parent}.type"),
                        ));
                    }
                    let (mut function, normalized) =
                        parse_function_tool(nested_object, &nested_parent)?;
                    let response_name = ResponseToolName {
                        namespace: Some(namespace.clone()),
                        name: function.name.clone(),
                    };
                    if !response_names.insert(response_name.clone()) {
                        return Err(ServerError::invalid_request(
                            format!(
                                "duplicate namespace function tool `{}.{}`",
                                namespace, function.name
                            ),
                            Some(&format!("{nested_parent}.name")),
                        ));
                    }
                    let alias = allocate_namespace_alias(
                        &namespace,
                        &function.name,
                        &mut occupied_chat_names,
                    );
                    function.name = alias.clone();
                    function.description = combine_namespace_description(
                        namespace_description.as_deref(),
                        function.description.as_deref(),
                    );
                    tool_names.insert(response_name, alias);
                    chat_tools.push(ChatTool {
                        tool_type: "function".to_string(),
                        function,
                    });
                    normalized_nested.push(normalized);
                }

                let mut normalized = Map::new();
                normalized.insert("type".to_string(), json!("namespace"));
                normalized.insert("name".to_string(), json!(namespace));
                if let Some(description) = namespace_description {
                    normalized.insert("description".to_string(), json!(description));
                }
                normalized.insert("tools".to_string(), Value::Array(normalized_nested));
                response_tools.push(Value::Object(normalized));
            }
            unsupported_type => {
                return Err(unsupported(
                    format!(
                        "tool type `{unsupported_type}` is not supported; function and namespace tools are supported"
                    ),
                    &type_param,
                ));
            }
        }
    }
    Ok((chat_tools, response_tools, tool_names))
}

fn parse_function_tool(
    object: &Map<String, Value>,
    parent: &str,
) -> std::result::Result<(ChatFunction, Value), ServerError> {
    reject_unknown_object_fields(
        object,
        &["type", "name", "description", "parameters", "strict"],
        parent,
    )?;
    let name = required_string(object, "name", &format!("{parent}.name"))?;
    let description = optional_string(object, "description", &format!("{parent}.description"))?;
    let parameters = object.get("parameters").cloned();
    if parameters.as_ref().is_some_and(|value| !value.is_object()) {
        return Err(ServerError::invalid_request(
            "function parameters must be a JSON object",
            Some(&format!("{parent}.parameters")),
        ));
    }
    let strict = optional_bool(object, "strict", &format!("{parent}.strict"))?;
    let function = ChatFunction {
        name: name.clone(),
        description: description.clone(),
        parameters: parameters.clone(),
        strict,
    };
    let mut normalized = Map::new();
    normalized.insert("type".to_string(), json!("function"));
    normalized.insert("name".to_string(), json!(name));
    if let Some(description) = description {
        normalized.insert("description".to_string(), json!(description));
    }
    if let Some(parameters) = parameters {
        normalized.insert("parameters".to_string(), parameters);
    }
    if let Some(strict) = strict {
        normalized.insert("strict".to_string(), json!(strict));
    }
    Ok((function, Value::Object(normalized)))
}

pub(super) fn allocate_namespace_alias(
    namespace: &str,
    name: &str,
    occupied: &mut HashSet<String>,
) -> String {
    let base = format!("{namespace}__{name}");
    if base.len() <= MAX_CHAT_TOOL_NAME_BYTES && occupied.insert(base.clone()) {
        return base;
    }

    for attempt in 0_u32.. {
        let hash = stable_namespace_alias_hash(namespace, name, attempt);
        let suffix = format!("__{hash:016x}");
        let prefix_budget = MAX_CHAT_TOOL_NAME_BYTES - suffix.len();
        let prefix_end = base
            .char_indices()
            .map(|(index, _)| index)
            .take_while(|index| *index <= prefix_budget)
            .last()
            .unwrap_or(0);
        let prefix_end = if base.len() <= prefix_budget {
            base.len()
        } else {
            prefix_end
        };
        let candidate = format!("{}{}", &base[..prefix_end], suffix);
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("u32 namespace alias space exhausted")
}

pub(super) fn stable_namespace_alias_hash(namespace: &str, name: &str, attempt: u32) -> u64 {
    // FNV-1a gives a stable, dependency-free identifier. The exact mapping is
    // request-local; the full (namespace, name) pair remains in ToolNameMap.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in namespace
        .bytes()
        .chain([0xff])
        .chain(name.bytes())
        .chain([0xfe])
        .chain(attempt.to_le_bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn combine_namespace_description(
    namespace_description: Option<&str>,
    function_description: Option<&str>,
) -> Option<String> {
    match (
        namespace_description.filter(|value| !value.is_empty()),
        function_description.filter(|value| !value.is_empty()),
    ) {
        (Some(namespace), Some(function)) => Some(format!("{namespace}\n\n{function}")),
        (Some(namespace), None) => Some(namespace.to_string()),
        (None, Some(function)) => Some(function.to_string()),
        (None, None) => None,
    }
}

pub(super) fn parse_tool_choice(
    choice: Option<&Value>,
    tool_names: &ToolNameMap,
) -> std::result::Result<(Option<ToolChoice>, Value), ServerError> {
    let Some(choice) = choice else {
        return Ok((None, json!("auto")));
    };
    match choice {
        Value::String(mode) if matches!(mode.as_str(), "auto" | "none" | "required") => {
            Ok((Some(ToolChoice::Mode(mode.clone())), json!(mode)))
        }
        Value::Object(object) => {
            let tool_type = required_string(object, "type", "tool_choice.type")?;
            if tool_type != "function" {
                return Err(unsupported(
                    "only function tool_choice objects are supported",
                    "tool_choice.type",
                ));
            }
            let name = required_string(object, "name", "tool_choice.name")?;
            let namespace = optional_string(object, "namespace", "tool_choice.namespace")?;
            if namespace.as_deref() == Some("") {
                return Err(ServerError::invalid_request(
                    "tool_choice.namespace must be a non-empty string",
                    Some("tool_choice.namespace"),
                ));
            }
            let chat_name = match namespace.as_deref() {
                Some(namespace) => tool_names
                    .chat_name(Some(namespace), &name)
                    .map(str::to_string)
                    .ok_or_else(|| {
                        ServerError::invalid_request(
                            format!(
                                "tool choice `{namespace}.{name}` is not present in request tools"
                            ),
                            Some("tool_choice.namespace"),
                        )
                    })?,
                None => tool_names
                    .chat_name(None, &name)
                    .unwrap_or(&name)
                    .to_string(),
            };
            let mut response_choice = Map::new();
            response_choice.insert("type".to_string(), json!("function"));
            response_choice.insert("name".to_string(), json!(name));
            if let Some(namespace) = namespace {
                response_choice.insert("namespace".to_string(), json!(namespace));
            }
            Ok((
                Some(ToolChoice::Function {
                    tool_type: "function".to_string(),
                    function: ToolChoiceFunction { name: chat_name },
                }),
                Value::Object(response_choice),
            ))
        }
        _ => Err(ServerError::invalid_request(
            "tool_choice must be auto, none, required, or a function selector",
            Some("tool_choice"),
        )),
    }
}
