//! CPU-only declaration of the actual fused F32 provider before native packaging.
use ferrum_kernels::native_ops::cutlass_fused_f32_attention::provider_catalog_declaration;
use ferrum_types::NativeOperatorProviderCatalog;
use std::{error::Error, fs::OpenOptions, io::Write};

fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(2..=3).contains(&args.len()) {
        return Err(
            "usage: cutlass_fused_f32_catalog_declaration <catalog-output> [base-catalog]".into(),
        );
    }
    let mut catalog = provider_catalog_declaration()?;
    if let Some(path) = args.get(2) {
        let mut base: NativeOperatorProviderCatalog =
            serde_json::from_slice(&std::fs::read(path)?)?;
        base.validate()?;
        if base.schema_version != catalog.schema_version || base.backend != catalog.backend {
            return Err("base catalog schema/backend differs from fused F32 declaration".into());
        }
        for row in &catalog.providers {
            base.providers.retain(|old| {
                old.operation_id != row.operation_id || old.provider_id != row.provider_id
            });
        }
        catalog.providers.extend(base.providers);
        catalog.providers.sort_by(|a, b| {
            (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id))
        });
    }
    let bytes = catalog.canonical_json_bytes()?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    println!(
        "source declaration only: sha256={} output={}",
        catalog.canonical_sha256()?,
        args[1]
    );
    Ok(())
}
