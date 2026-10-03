use ferrum_bench_core::required_query_audit::{
    audit_required_queries, compare_query_universes, read_universe_comparison_input, AuditOptions,
};
use std::{
    fs::File,
    io::{self, BufReader, Write},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or(
        "usage: required_query_audit <JSONL path|-> [--through-transaction N] [--before-ordinal N] [--universes PATH]",
    )?;
    let mut options = AuditOptions::default();
    let mut universes = None;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing option value")?;
        match flag.as_str() {
            "--through-transaction" => options.through_transaction = Some(value.parse()?),
            "--before-ordinal" => options.before_ordinal = Some(value.parse()?),
            "--universes" => universes = Some(value),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if input == "-" && universes.is_some() {
        return Err("--universes requires an original JSONL file for two bounded passes".into());
    }
    let reader: Box<dyn io::BufRead> = if input == "-" {
        Box::new(BufReader::new(io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(&input)?))
    };
    let report = audit_required_queries(reader, options)?;
    let complete = report.integrity.trace_content_complete;
    let mut output = io::stdout().lock();
    if let Some(path) = universes {
        let declaration = read_universe_comparison_input(File::open(path)?)?;
        let comparison = if complete {
            Some(compare_query_universes(
                BufReader::new(File::open(&input)?),
                &report,
                declaration,
            )?)
        } else {
            None
        };
        serde_json::to_writer(
            &mut output,
            &serde_json::json!({
                "audit": report,
                "universe_comparison": comparison,
                "comparison_status": if complete { "completed_necessary_conditions_only" } else { "skipped_incomplete_trace" }
            }),
        )?;
    } else {
        serde_json::to_writer(&mut output, &report)?;
    }
    writeln!(output)?;
    if !complete {
        std::process::exit(2);
    }
    Ok(())
}
