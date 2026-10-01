use ferrum_bench_core::required_query_audit::{audit_required_queries, AuditOptions};
use std::{
    fs::File,
    io::{self, BufReader, Write},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args.next().ok_or(
        "usage: required_query_audit <JSONL path|-> [--through-transaction N] [--before-ordinal N]",
    )?;
    let mut options = AuditOptions::default();
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("missing option value")?.parse::<u64>()?;
        match flag.as_str() {
            "--through-transaction" => options.through_transaction = Some(value),
            "--before-ordinal" => options.before_ordinal = Some(value),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let reader: Box<dyn io::BufRead> = if input == "-" {
        Box::new(BufReader::new(io::stdin()))
    } else {
        Box::new(BufReader::new(File::open(&input)?))
    };
    let report = audit_required_queries(reader, options)?;
    let complete = report.integrity.trace_content_complete;
    let mut output = io::stdout().lock();
    serde_json::to_writer(&mut output, &report)?;
    writeln!(output)?;
    if !complete {
        std::process::exit(2);
    }
    Ok(())
}
