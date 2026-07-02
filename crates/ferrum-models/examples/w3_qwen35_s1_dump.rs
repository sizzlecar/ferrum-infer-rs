use std::path::PathBuf;

use ferrum_models::qwen35_s1::write_qwen35_s1_dump;

fn parse_path(args: &[String], flag: &str) -> Result<PathBuf, String> {
    let Some(idx) = args.iter().position(|arg| arg == flag) else {
        return Err(format!(
            "usage: w3_qwen35_s1_dump --model-dir <hf-snapshot> --hf-dump <hf-dump-dir> --out <dir>"
        ));
    };
    let Some(value) = args.get(idx + 1) else {
        return Err(format!("{flag} requires a value"));
    };
    Ok(PathBuf::from(value))
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let result = (|| {
        let model_dir = parse_path(&args, "--model-dir")?;
        let hf_dump = parse_path(&args, "--hf-dump")?;
        let out = parse_path(&args, "--out")?;
        write_qwen35_s1_dump(&model_dir, &hf_dump, &out, "ferrum-models-w3-qwen35-s1")?;
        Ok::<(), String>(())
    })();
    if let Err(err) = result {
        eprintln!("W3 QWEN35 FERRUM LAYER DUMP FAIL: {err}");
        std::process::exit(1);
    }
}
