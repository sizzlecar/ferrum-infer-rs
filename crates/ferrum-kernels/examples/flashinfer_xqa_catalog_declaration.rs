//! Declare the source-bound GPT-OSS provider before building its native artifact.
use ferrum_kernels::native_ops::flashinfer_xqa::provider_catalog_declaration;
use ferrum_types::NativeOperatorProviderCatalog;
use std::{error::Error, fs::OpenOptions, io::Write};

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err(
            "usage: flashinfer_xqa_catalog_declaration <base-catalog> <catalog-output>".into(),
        );
    }
    let mut base: NativeOperatorProviderCatalog =
        serde_json::from_slice(&std::fs::read(&args[1])?)?;
    base.validate()?;
    let declaration = provider_catalog_declaration()?;
    if base.schema_version != declaration.schema_version || base.backend != declaration.backend {
        return Err("base catalog schema/backend differs from XQA declaration".into());
    }
    for row in declaration.providers {
        base.providers.retain(|old| {
            old.operation_id != row.operation_id || old.provider_id != row.provider_id
        });
        base.providers.push(row);
    }
    base.providers
        .sort_by(|a, b| (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id)));
    let bytes = base.canonical_json_bytes()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!(
        "source declaration only: sha256={} output={}",
        base.canonical_sha256()?,
        args[2]
    );
    Ok(())
}
