use std::path::PathBuf;

use ferrum_models::deltanet_s1::{write_w3_deltanet_s1_dump, W3DeltaNetS1Shape};

fn parse_value(args: &[String], flag: &str, default: usize) -> Result<usize, String> {
    let Some(idx) = args.iter().position(|arg| arg == flag) else {
        return Ok(default);
    };
    let Some(value) = args.get(idx + 1) else {
        return Err(format!("{flag} requires a value"));
    };
    value
        .parse::<usize>()
        .map_err(|err| format!("{flag} value {value:?} is invalid: {err}"))
}

fn parse_seed(args: &[String], default: u32) -> Result<u32, String> {
    let Some(idx) = args.iter().position(|arg| arg == "--seed") else {
        return Ok(default);
    };
    let Some(value) = args.get(idx + 1) else {
        return Err("--seed requires a value".to_string());
    };
    value
        .parse::<u32>()
        .map_err(|err| format!("--seed value {value:?} is invalid: {err}"))
}

fn parse_out(args: &[String]) -> Result<PathBuf, String> {
    let Some(idx) = args.iter().position(|arg| arg == "--out") else {
        return Err("usage: w3_deltanet_s1_dump --out <dir> [shape flags]".to_string());
    };
    let Some(value) = args.get(idx + 1) else {
        return Err("--out requires a value".to_string());
    };
    Ok(PathBuf::from(value))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = (|| {
        let default_shape = W3DeltaNetS1Shape::default();
        let out = parse_out(&args)?;
        let shape = W3DeltaNetS1Shape {
            tokens: parse_value(&args, "--tokens", default_shape.tokens)?,
            hidden_dim: parse_value(&args, "--hidden-dim", default_shape.hidden_dim)?,
            heads: parse_value(&args, "--heads", default_shape.heads)?,
            key_dim: parse_value(&args, "--key-dim", default_shape.key_dim)?,
            value_dim: parse_value(&args, "--value-dim", default_shape.value_dim)?,
            experts: parse_value(&args, "--experts", default_shape.experts)?,
            top_k: parse_value(&args, "--top-k", default_shape.top_k)?,
            expert_hidden_dim: parse_value(
                &args,
                "--expert-hidden-dim",
                default_shape.expert_hidden_dim,
            )?,
        };
        let seed = parse_seed(&args, 9271)?;
        write_w3_deltanet_s1_dump(&out, shape, seed, "ferrum-models-w3-deltanet-s1")?;
        println!("W3 DELTANET S1 FERRUM DUMP PASS: {}", out.display());
        Ok::<(), String>(())
    })();
    if let Err(err) = result {
        eprintln!("W3 DELTANET S1 FERRUM DUMP FAIL: {err}");
        std::process::exit(1);
    }
}
