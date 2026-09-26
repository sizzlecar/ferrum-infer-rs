use super::{LaunchConfig, Workspace};

pub(super) const PACK: &str = "vnext_q4_stream_pack";
pub(super) const PROJECT: &str = "vnext_q4_stream_mmq";
pub(super) const FIXUP: &str = "vnext_q4_stream_fixup";

pub(super) fn configs(
    rows: u32,
    hidden: u32,
    intermediate: u32,
    layout: Workspace,
) -> Result<[LaunchConfig; 3], String> {
    let groups = rows
        .checked_mul(hidden / 32)
        .ok_or("Stream-MMQ pack extent overflow")?;
    let values = rows
        .checked_mul(intermediate)
        .ok_or("Stream-MMQ output extent overflow")?;
    Ok([
        LaunchConfig {
            grid_dim: (groups.div_ceil(8), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        },
        LaunchConfig {
            grid_dim: (layout.ctas, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: layout.precision.shared_bytes(),
        },
        LaunchConfig::for_num_elems(values),
    ])
}
