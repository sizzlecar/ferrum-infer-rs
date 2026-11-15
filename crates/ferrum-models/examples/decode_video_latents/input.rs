use super::*;
pub(super) fn load(
    c: &Configuration,
    host: &HostMemoryAccount,
) -> anyhow::Result<ManagedHostBytes> {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, c.input_bytes()?)?;
    match &c.input {
        super::super::config::LatentInput::File { path } => {
            let mut file = std::fs::File::open(path)?;
            anyhow::ensure!(
                file.metadata()?.len() == c.input_bytes()? as u64,
                "latent file extent differs from explicit F32 shape/layout"
            );
            file.read_exact(bytes.as_mut())?;
        }
        super::super::config::LatentInput::DiagnosticZeros => {}
        super::super::config::LatentInput::DiagnosticDyadic => {
            for (index, word) in bytes.as_mut().chunks_exact_mut(4).enumerate() {
                let magnitude = (1 + (index % 31)) as f32 / 64f32;
                let value = if (index / 31) % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
                word.copy_from_slice(&value.to_le_bytes());
            }
        }
    }
    c.latent_layout
        .managed_ncthw(c.latent_shape, &bytes.freeze(), host, c.host_bytes)
}
