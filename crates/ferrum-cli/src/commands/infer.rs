//! Inference command implementation

use crate::{chat_template, client::FerrumClient, config::CliConfig, output::OutputFormat};
use chrono::Utc;
use clap::Args;
use colored::*;
use ferrum_interfaces::engine::InferenceEngine;
use ferrum_models::ModelSourceResolver;
use ferrum_server::openai::{ChatCompletionsRequest, ChatMessage, MessageRole};
use ferrum_types::{InferenceRequest, Priority, Result, SamplingParams};
use futures::StreamExt;
use std::collections::HashMap;
use std::io::{self, Write};
use uuid::Uuid;

use tracing::{info, instrument, warn};

#[derive(Args, Debug)]
pub struct InferCommand {
    /// Prompt text
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// Model to use
    #[arg(short, long)]
    pub model: Option<String>,

    /// Maximum tokens to generate
    #[arg(long, default_value = "100")]
    pub max_tokens: u32,

    /// Temperature for sampling
    #[arg(long, default_value = "0.7")]
    pub temperature: f32,

    /// Top-p for nucleus sampling
    #[arg(long, default_value = "0.9")]
    pub top_p: f32,

    /// Enable streaming output
    #[arg(long)]
    pub stream: bool,

    /// Input file containing prompt
    #[arg(long)]
    pub input_file: Option<String>,

    /// Output file for results
    #[arg(long)]
    pub output_file: Option<String>,

    /// Interactive mode for continuous conversation (enabled by default)
    #[arg(short, long)]
    pub interactive: bool,

    /// One-shot mode (single inference, then exit)
    #[arg(long)]
    pub once: bool,

    /// Server URL
    #[arg(long)]
    pub url: Option<String>,

    /// Backend to use (cpu, metal, cuda:0)
    #[arg(long, default_value = "auto")]
    pub backend: String,
}

#[instrument(skip(config), fields(model = %cmd.model.as_ref().unwrap_or(&"unknown".to_string()), stream = cmd.stream))]
pub async fn execute(cmd: InferCommand, config: CliConfig, _format: OutputFormat) -> Result<()> {
    info!("Starting CLI inference command");
    println!("{} Running inference...", "🧠".bright_blue());

    // Determine if we should enter interactive mode
    // Default to interactive unless --once is specified or using --url (remote server)
    let should_be_interactive = !cmd.once && cmd.url.is_none();

    if should_be_interactive && cmd.input_file.is_none() {
        // Enter interactive conversation mode
        return run_interactive_mode(cmd, config).await;
    }

    // One-shot mode: execute single inference
    let prompt = if let Some(prompt) = cmd.prompt.clone() {
        prompt
    } else if let Some(file_path) = cmd.input_file.clone() {
        tokio::fs::read_to_string(&file_path).await.map_err(|e| {
            ferrum_types::FerrumError::io_str(format!("Failed to read input file: {}", e))
        })?
    } else {
        return Err(ferrum_types::FerrumError::invalid_request(
            "Please provide --prompt, --input-file, or enter interactive mode".to_string(),
        ));
    };

    // Determine model
    let model = cmd
        .model
        .clone()
        .or(config.models.default_model.clone())
        .ok_or_else(|| {
            ferrum_types::FerrumError::invalid_request("No model specified".to_string())
        })?;

    println!("Model: {}", model.cyan());
    println!("Prompt: {}", prompt.trim().bright_white());
    println!("Max tokens: {}", cmd.max_tokens.to_string().cyan());
    println!("Temperature: {}", cmd.temperature.to_string().cyan());
    println!("Top-p: {}", cmd.top_p.to_string().cyan());

    if cmd.stream {
        println!("{} Streaming enabled", "📡".yellow());
    }

    // Create inference request
    let sampling_params = SamplingParams {
        max_tokens: cmd.max_tokens as usize,
        temperature: cmd.temperature,
        top_p: cmd.top_p,
        ..Default::default()
    };

    let request = InferenceRequest {
        id: ferrum_types::RequestId(uuid::Uuid::new_v4()),
        model_id: ferrum_types::ModelId(model),
        prompt,
        sampling_params,
        stream: cmd.stream,
        priority: Priority::Normal,
        client_id: None,
        session_id: None,
        created_at: Utc::now(),
        metadata: HashMap::new(),
    };

    // Run inference
    let result = if let Some(url) = cmd.url {
        // Remote mode: use HTTP client
        run_remote_inference(&request, &url).await?
    } else {
        // Local mode: use direct backend
        run_local_inference(&request, &cmd).await?
    };

    // Display or save output
    if let Some(output_path) = cmd.output_file {
        let output_content = result.text.clone();

        tokio::fs::write(&output_path, output_content)
            .await
            .map_err(|e| {
                ferrum_types::FerrumError::io_str(format!("Failed to write output file: {}", e))
            })?;

        println!("{} Output saved to: {}", "💾".green(), output_path.cyan());
    } else {
        // Print to stdout
        println!("\n{}", "📝 Generated Text:".bright_green().bold());
        if result.text.trim().is_empty() {
            println!("{}", "(空输出或仅包含空格)".dimmed());
        } else {
            println!("{}", result.text.bright_white());
        }

        println!("\n{}", "📊 Statistics:".bright_blue());
        println!(
            "  Tokens generated: {}",
            result.usage.completion_tokens.to_string().cyan()
        );
        println!(
            "  Prompt tokens: {}",
            result.usage.prompt_tokens.to_string().cyan()
        );
        println!(
            "  Total tokens: {}",
            result.usage.total_tokens.to_string().cyan()
        );
        println!("  Finish reason: {:?}", result.finish_reason);
        println!("  Total time: {}ms", result.latency_ms.to_string().yellow());
    }

    Ok(())
}

/// Run inference using local backend with enhanced model loading
#[instrument(skip(request, cmd), fields(request_id = %request.id.0, model_id = %request.model_id.0))]
async fn run_local_inference(
    request: &InferenceRequest,
    cmd: &InferCommand,
) -> Result<ferrum_types::InferenceResponse> {
    info!("Starting local inference with Candle backend");
    println!("{} Starting local inference...", "🔥".bright_yellow());

    // Resolve model using enhanced model maintenance system
    let model_id = &request.model_id.0;
    println!(
        "{} Resolving model: {}",
        "🔍".bright_blue(),
        model_id.cyan()
    );

    let registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let resolved_id = registry.resolve_model_id(model_id);

    if resolved_id != *model_id {
        println!(
            "{} Resolved alias '{}' to: {}",
            "🔗".bright_blue(),
            model_id.cyan(),
            resolved_id.yellow()
        );
    }

    // Create model source resolver
    let source_config = ferrum_models::ModelSourceConfig::default();
    let resolver = ferrum_models::DefaultModelSourceResolver::new(source_config);

    // Resolve model source
    let source = match resolver.resolve(&resolved_id, None).await {
        Ok(source) => {
            println!(
                "{} Model resolved: {}",
                "✅".bright_green(),
                source.local_path.display()
            );
            source
        }
        Err(e) => {
            println!(
                "{} Failed to resolve model '{}': {}",
                "❌".bright_red(),
                resolved_id,
                e
            );
            println!("\nTips:");
            println!(
                "  - Use 'ferrum models --download {}' to download the model",
                resolved_id
            );
            println!("  - Check if the model exists locally with 'ferrum models --list'");
            return Err(e);
        }
    };

    // Load model configuration
    println!("{} Loading model configuration...", "⚙️".bright_blue());
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_config = match config_manager.load_from_source(&source).await {
        Ok(config) => {
            println!(
                "{} Configuration loaded: {} ({})",
                "✅".bright_green(),
                format!("{:?}", config.architecture).yellow(),
                format!("{} params", config.vocab_size).cyan()
            );
            config
        }
        Err(e) => {
            println!("{} Failed to load configuration: {}", "❌".bright_red(), e);
            return Err(e);
        }
    };

    // Create engine configuration with resolved model information
    let device = match cmd.backend.as_str() {
        "auto" => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                if cfg!(feature = "metal") {
                    println!(
                        "{} Auto-detected Metal backend for Apple GPU",
                        "🔥".yellow()
                    );
                    ferrum_types::Device::Metal
                } else {
                    println!("{} Auto-detected CPU backend", "💻".blue());
                    ferrum_types::Device::CPU
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                println!("{} Auto-detected CPU backend", "💻".blue());
                ferrum_types::Device::CPU
            }
        }
        "cpu" => ferrum_types::Device::CPU,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => ferrum_types::Device::Metal,
        backend if backend.starts_with("cuda:") => {
            let device_id = backend[5..].parse().unwrap_or(0);
            ferrum_types::Device::CUDA(device_id)
        }
        backend => {
            println!("{} Using {} backend", "⚙️".blue(), backend.cyan());
            ferrum_types::Device::CPU
        }
    };

    // 设置模型路径环境变量，以便引擎工厂加载真实的 tokenizer 和模型
    std::env::set_var(
        "FERRUM_MODEL_PATH",
        source.local_path.to_string_lossy().to_string(),
    );

    let engine_config = ferrum_engine::simple_engine_config(resolved_id.clone(), device);

    println!("{} Initializing inference engine...", "⚙️".yellow());
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;
    println!("{} Engine initialized successfully", "✅".green());

    println!("{} Generating...", "⚡".bright_magenta());

    let start_time = std::time::Instant::now();

    if request.stream {
        print!("Output: ");
        io::stdout().flush().unwrap();

        // Use streaming inference
        let mut stream = engine.infer_stream(request.clone()).await?;
        let mut full_text = String::new();
        let mut token_count = 0;

        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    // 安全地处理可能的编码问题
                    if !chunk.text.is_empty() {
                        print!("{}", chunk.text);
                        io::stdout().flush().unwrap();
                        full_text.push_str(&chunk.text);
                    }

                    if chunk.token.is_some() {
                        token_count += 1;
                    }

                    if chunk.finish_reason.is_some() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("\n{} Stream error: {}", "❌".red(), e);
                    break;
                }
            }
        }

        println!("\n{} Generation completed", "🏁".green());

        let duration = start_time.elapsed();
        Ok(ferrum_types::InferenceResponse {
            request_id: request.id.clone(),
            text: full_text,
            tokens: vec![], // Tokens not available in streaming mode
            finish_reason: ferrum_types::FinishReason::Length,
            usage: ferrum_types::TokenUsage {
                prompt_tokens: 0, // TODO: get from engine
                completion_tokens: token_count,
                total_tokens: token_count,
            },
            latency_ms: duration.as_millis() as u64,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        })
    } else {
        // Use non-streaming inference
        let response = engine.infer(request.clone()).await?;
        println!("{} Generation completed", "🏁".green());
        Ok(response)
    }
}

/// Run inference using HTTP client (remote mode)
#[instrument(skip(request), fields(request_id = %request.id.0, model_id = %request.model_id.0, url = url))]
async fn run_remote_inference(
    request: &InferenceRequest,
    url: &str,
) -> Result<ferrum_types::InferenceResponse> {
    info!("Starting remote inference to server");
    println!(
        "{} Starting remote inference to: {}",
        "🌐".bright_cyan(),
        url
    );

    // Create temporary config for client
    let mut config = CliConfig::default();
    config.client.base_url = url.to_string();

    let client = FerrumClient::new(config)?;

    // Convert to OpenAI chat completions format
    let chat_request = ChatCompletionsRequest {
        model: request.model_id.0.clone(),
        messages: vec![ChatMessage {
            role: MessageRole::User,
            content: request.prompt.clone(),
            name: None,
        }],
        max_tokens: Some(request.sampling_params.max_tokens as u32),
        n: Some(1),
        seed: request.sampling_params.seed,
        temperature: Some(request.sampling_params.temperature),
        top_p: Some(request.sampling_params.top_p),
        stream: Some(request.stream),
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        user: None,
    };

    if request.stream {
        info!("Starting streaming remote inference");
        println!(
            "{} Streaming not yet implemented for remote mode",
            "⚠️".yellow()
        );
        warn!("Falling back to non-streaming mode for remote calls");
        // For MVP, fall back to non-streaming for remote calls
    }

    let chat_response = client.chat_completions(&chat_request).await?;

    // Convert response
    let generated_text = chat_response
        .choices
        .first()
        .and_then(|choice| choice.message.as_ref())
        .map(|msg| msg.content.clone())
        .unwrap_or_default();

    // Extract token usage info for usage statistics

    Ok(ferrum_types::InferenceResponse {
        request_id: request.id.clone(),
        text: generated_text,
        tokens: Vec::new(), // Remote doesn't return tokens
        finish_reason: ferrum_types::FinishReason::Length, // TODO: Parse from response
        usage: chat_response
            .usage
            .map(|u| ferrum_types::TokenUsage {
                prompt_tokens: u.prompt_tokens as usize,
                completion_tokens: u.completion_tokens as usize,
                total_tokens: u.total_tokens as usize,
            })
            .unwrap_or(ferrum_types::TokenUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            }),
        metadata: HashMap::new(),
        latency_ms: 0, // TODO: Calculate latency
        created_at: Utc::now(),
    })
}

/// Interactive conversation mode (like ollama)
async fn run_interactive_mode(cmd: InferCommand, config: CliConfig) -> Result<()> {
    println!(
        "{}",
        "🚀 Starting interactive conversation mode"
            .bright_green()
            .bold()
    );
    println!(
        "{}",
        "Type your message and press Enter. Use 'exit' or Ctrl+C to quit.".dimmed()
    );

    // Initialize engine once with enhanced model resolution
    let model_name = cmd
        .model
        .clone()
        .or(config.models.default_model.clone())
        .unwrap_or("TinyLlama-1.1B-Chat-v1.0".to_string());

    println!(
        "{} Resolving model for interactive mode: {}",
        "🔍".bright_blue(),
        model_name.cyan()
    );

    let registry = ferrum_models::DefaultModelRegistry::with_defaults();
    let resolved_id = registry.resolve_model_id(&model_name);

    if resolved_id != model_name {
        println!(
            "{} Resolved alias '{}' to: {}",
            "🔗".bright_blue(),
            model_name.cyan(),
            resolved_id.yellow()
        );
    }

    // Create model source resolver
    let source_config = ferrum_models::ModelSourceConfig::default();
    let resolver = ferrum_models::DefaultModelSourceResolver::new(source_config);

    // Resolve model source
    let source = match resolver.resolve(&resolved_id, None).await {
        Ok(source) => {
            println!(
                "{} Model resolved: {}",
                "✅".bright_green(),
                source.local_path.display()
            );
            source
        }
        Err(e) => {
            println!(
                "{} Failed to resolve model '{}': {}",
                "❌".bright_red(),
                resolved_id,
                e
            );
            println!(
                "\nTip: Use 'ferrum models --download {}' to download the model",
                resolved_id
            );
            return Err(e);
        }
    };

    // Load model configuration for better engine setup
    let mut config_manager = ferrum_models::ConfigManager::new();
    let model_config = config_manager
        .load_from_source(&source)
        .await
        .unwrap_or_else(|e| {
            warn!("Failed to load model config, using defaults: {}", e);
            ferrum_models::ModelDefinition {
                architecture: ferrum_models::Architecture::Llama,
                hidden_size: 4096,
                intermediate_size: 11008,
                vocab_size: 32000,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: None,
                max_position_embeddings: 2048,
                rope_theta: Some(10000.0),
                rope_scaling: None,
                norm_type: ferrum_models::NormType::RMSNorm,
                norm_eps: 1e-6,
                attention_config: ferrum_models::AttentionConfig {
                    attention_bias: false,
                    sliding_window: None,
                },
                activation: ferrum_models::Activation::SiLU,
                extra_params: serde_json::Value::Object(serde_json::Map::new()),
            }
        });

    let device = match cmd.backend.as_str() {
        "auto" => {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            {
                if cfg!(feature = "metal") {
                    println!(
                        "{} Auto-detected Metal backend for Apple GPU",
                        "🔥".yellow()
                    );
                    ferrum_types::Device::Metal
                } else {
                    println!("{} Auto-detected CPU backend", "💻".blue());
                    ferrum_types::Device::CPU
                }
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            {
                println!("{} Auto-detected CPU backend", "💻".blue());
                ferrum_types::Device::CPU
            }
        }
        "cpu" => ferrum_types::Device::CPU,
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        "metal" => ferrum_types::Device::Metal,
        backend if backend.starts_with("cuda:") => {
            let device_id = backend[5..].parse().unwrap_or(0);
            ferrum_types::Device::CUDA(device_id)
        }
        backend => {
            println!("{} Using {} backend", "⚙️".blue(), backend.cyan());
            ferrum_types::Device::CPU
        }
    };

    // 设置模型路径环境变量，以便引擎工厂加载真实的 tokenizer 和模型
    std::env::set_var(
        "FERRUM_MODEL_PATH",
        source.local_path.to_string_lossy().to_string(),
    );

    let engine_config = ferrum_engine::simple_engine_config(resolved_id.clone(), device);

    println!("{} Initializing inference engine...", "⚙️".yellow());
    let engine = ferrum_engine::create_mvp_engine(engine_config).await?;
    println!("{} Engine ready for conversation!", "✅".green());

    let mut conversation_history = Vec::new();
    let mut conversation_turn = 1;

    loop {
        // 显示提示符
        print!("\n{} ", ">>>".bright_blue().bold());
        io::stdout().flush().unwrap();

        // 读取用户输入
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {
                let input = input.trim();

                // 退出命令
                if input.is_empty() {
                    continue;
                }
                if input == "exit" || input == "quit" || input == "bye" {
                    println!("{} Goodbye!", "👋".bright_yellow());
                    break;
                }

                // 先添加当前用户输入到历史
                conversation_history.push(format!("User: {}", input));

                // 构建包含完整历史的prompt - 使用 chat template
                let full_prompt = {
                    // 将历史转换为 (role, content) 格式
                    let mut history_pairs = Vec::new();
                    for (i, msg) in conversation_history.iter().enumerate() {
                        let role = if i % 2 == 0 { "user" } else { "assistant" };
                        let content = msg
                            .trim_start_matches("User: ")
                            .trim_start_matches("Assistant: ");
                        if i < conversation_history.len() - 1 {
                            // 排除当前用户输入
                            history_pairs.push((role.to_string(), content.to_string()));
                        }
                    }

                    // 使用 chat template 格式化
                    chat_template::auto_format_prompt(&resolved_id, &history_pairs, &input)
                };

                // Debug: 显示完整prompt以便调试
                println!(
                    "{} Prompt context: {} history entries",
                    "🔍".dimmed(),
                    conversation_history.len().to_string().dimmed()
                );

                // 创建推理请求
                let request = ferrum_types::InferenceRequest {
                    id: ferrum_types::RequestId(Uuid::new_v4()),
                    model_id: ferrum_types::ModelId(
                        cmd.model
                            .clone()
                            .or(config.models.default_model.clone())
                            .unwrap_or("dummy".to_string()),
                    ),
                    prompt: full_prompt,
                    created_at: Utc::now(),
                    sampling_params: SamplingParams {
                        max_tokens: cmd.max_tokens as usize,
                        temperature: cmd.temperature,
                        top_p: cmd.top_p,
                        top_k: None,
                        frequency_penalty: 0.0,
                        presence_penalty: 0.0,
                        repetition_penalty: 1.0,
                        stop_sequences: vec![
                            "User:".to_string(),
                            "\nUser:".to_string(),
                            ">>> ".to_string(),
                            "\n\n".to_string(),
                        ],
                        seed: None,
                        min_p: None,
                        tfs: None,
                        typical_p: None,
                        mirostat: None,
                    },
                    priority: Priority::Normal,
                    stream: true, // 始终使用流式输出
                    client_id: None,
                    session_id: None,
                    metadata: HashMap::new(),
                };

                // Add missing fields to sampling_params
                let mut sampling_params = request.sampling_params.clone();
                sampling_params.min_p = None;
                sampling_params.tfs = None;
                sampling_params.typical_p = None;
                sampling_params.mirostat = None;
                let request = ferrum_types::InferenceRequest {
                    sampling_params,
                    ..request
                };

                println!(
                    "\n{} Turn {}",
                    "🤖".bright_cyan(),
                    conversation_turn.to_string().yellow()
                );
                print!("{}", "Assistant:".bright_green());
                io::stdout().flush().unwrap();

                // 流式生成回复
                let mut stream = engine.infer_stream(request).await?;
                let mut response_text = String::new();
                let mut token_count = 0;
                let start_time = std::time::Instant::now();

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(chunk) => {
                            if !chunk.text.is_empty() {
                                print!("{}", chunk.text);
                                io::stdout().flush().unwrap();
                                response_text.push_str(&chunk.text);
                            }

                            if chunk.token.is_some() {
                                token_count += 1;
                            }

                            if chunk.finish_reason.is_some() {
                                break;
                            }
                        }
                        Err(e) => {
                            eprintln!("\n{} Error: {}", "❌".red(), e);
                            break;
                        }
                    }
                }

                let duration = start_time.elapsed();

                // 添加助手回复到历史
                if !response_text.is_empty() {
                    let cleaned_response = response_text
                        .trim()
                        .strip_suffix("User:")
                        .unwrap_or(response_text.trim())
                        .strip_suffix("\nUser:")
                        .unwrap_or(response_text.trim())
                        .strip_suffix("User")
                        .unwrap_or(response_text.trim())
                        .trim();
                    if !cleaned_response.is_empty() {
                        conversation_history.push(format!("Assistant: {}", cleaned_response));
                    }
                }

                // 显示统计信息
                println!(
                    "\n{} {} tokens in {}ms",
                    "📊".dimmed(),
                    token_count.to_string().dimmed(),
                    duration.as_millis().to_string().dimmed()
                );

                conversation_turn += 1;

                // 限制历史长度避免context过长（保持最近5轮对话）
                if conversation_history.len() > 10 {
                    conversation_history.drain(0..2);
                }

                // Debug: 显示当前对话历史（可以帮助调试上下文问题）
                if conversation_history.len() > 2 {
                    println!(
                        "{} Context: {} turns in history",
                        "🧠".dimmed(),
                        (conversation_history.len() / 2).to_string().dimmed()
                    );
                }
            }
            Err(e) => {
                eprintln!("{} Failed to read input: {}", "❌".red(), e);
                break;
            }
        }
    }

    Ok(())
}
