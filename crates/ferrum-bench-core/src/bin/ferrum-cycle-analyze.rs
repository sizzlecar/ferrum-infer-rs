//! Analyze persisted host-cycle evidence; it neither starts nor measures inference.
use ferrum_bench_core::decode_cycle::analyze_profile_jsonl;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: ferrum-cycle-analyze PROFILE.jsonl [PHYSICAL_WIDTH]")?;
    let width = args
        .next()
        .map(|width| {
            width
                .to_str()
                .ok_or("physical width must be UTF-8")?
                .parse::<usize>()
                .map_err(|_| "physical width must be a positive integer")
        })
        .transpose()?;
    if args.next().is_some() || width == Some(0) {
        return Err("usage: ferrum-cycle-analyze PROFILE.jsonl [PHYSICAL_WIDTH > 0]".into());
    }
    let input = std::fs::read_to_string(path)?;
    let analysis = analyze_profile_jsonl(&input, width)?;
    let mut output = std::io::stdout().lock();
    serde_json::to_writer(&mut output, &analysis)?;
    writeln!(output)?;
    Ok(())
}
