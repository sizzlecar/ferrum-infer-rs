//! Source text must survive the public OpenAI content-parts input and the
//! model's real whitespace handling, not just a synthetic concatenation.

use ferrum_server::chat_template::{
    render_chat_prompt_with_model_template_options, ChatTemplateOptions, ModelChatTemplate,
};
use ferrum_server::openai::ChatMessage;
use serde_json::json;

const QWEN_TEMPLATE: &str =
    include_str!("fixtures/chat_template/Qwen__Qwen3.5-35B-A3B/template.jinja");

#[test]
fn qwen_tool_text_parts_preserve_fenced_source_boundaries() {
    let template = ModelChatTemplate::new(QWEN_TEMPLATE, "qwen3.5-text-parts");
    let mut direct_env = minijinja::Environment::new();
    direct_env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    direct_env.set_trim_blocks(true);
    direct_env.set_lstrip_blocks(true);
    direct_env.add_template("qwen", QWEN_TEMPLATE).unwrap();
    let cases = [
        (
            "    let path = \"C:\\src\\code.rs\";\n\tlet x = 1;\n",
            "```",
        ),
        ("    // 雪豹\r\n\tlet x = \"\\n\";\r\n", "```"),
        ("    // no final newline", "```"),
        ("    // trailing whitespace  \t\r", "```"),
        ("\t// ``` and ```` inside source\n", "`````"),
        ("", "```"),
    ];
    for (source, fence) in cases {
        let final_newline = source.ends_with('\n');
        let label = if final_newline { "yes" } else { "no" };
        let separator = if final_newline { "" } else { "\n" };
        let header = format!("Text field \"content\" (final newline: {label})\n{fence}text\n");
        let source_part = format!("{header}{source}{separator}{fence}\n");
        let metadata = "Tool result metadata:\n{\"is_error\":false,\"result\":{\"eof\":true}}\n";
        let path_part = "Text field \"path\" (final newline: no)\n```text\nsrc/main.rs\n```\n";
        let request = json!([
            {"role":"user","content":"Inspect the returned source."},
            {"role":"tool","tool_call_id":"read-source","content":[
                {"type":"text","text":metadata},
                {"type":"text","text":source_part},
                {"type":"text","text":path_part}
            ]}
        ]);
        // The original HF template concatenates arrays without separators;
        // Ferrum's public content-parts deserializer inserts a newline.
        // Both consumers must close each fence on its own line while keeping
        // every byte inside the source field, including trailing whitespace.
        let direct = direct_env
            .get_template("qwen")
            .unwrap()
            .render(json!({
                "messages": request,
                "tools": [],
                "add_generation_prompt": true,
                "enable_thinking": false
            }))
            .unwrap();
        let messages: Vec<ChatMessage> = serde_json::from_value(request).unwrap();
        let rendered = render_chat_prompt_with_model_template_options(
            &messages,
            "test-model",
            Some(&template),
            &ChatTemplateOptions {
                enable_thinking: Some(false),
                ..Default::default()
            },
        )
        .unwrap();
        for (prompt, between_parts) in [(direct, ""), (rendered, "\n")] {
            let (_, tool_response) = prompt.split_once("<tool_response>\n").unwrap();
            let (body, _) = tool_response.split_once("\n</tool_response>").unwrap();
            assert_eq!(
                body,
                [metadata, &source_part, path_part]
                    .join(between_parts)
                    .trim_end()
            );
            let rendered_source = body
                .strip_prefix(&format!("{metadata}{between_parts}{header}"))
                .unwrap()
                .strip_suffix(&format!(
                    "{separator}{fence}\n{between_parts}{}",
                    path_part.trim_end()
                ))
                .unwrap();
            assert_eq!(rendered_source.as_bytes(), source.as_bytes());
        }
    }
}
