//! A controller receipt alone cannot authorize reuse: bench-core checks the
//! exact chronological original history and successful drained origin.
use super::*;

pub(super) fn authorize(
    cmd: &BenchServeCommand,
    search: &CapacitySearch,
) -> Result<Option<AuthorizedCapacitySessionBlock>> {
    let Some(path) = &cmd.capacity.capacity_session_receipt else {
        if cmd.capacity.capacity_reuse_warmup_from.is_some() {
            return Err(err("warmup reuse requires a controller session"));
        }
        return Ok(None);
    };
    let block: CapacitySessionBlock = read_json(path)?;
    if block.session.endpoint != cmd.base_url {
        return Err(err("capacity session endpoint differs from HTTP target"));
    }
    search
        .authorize_session_block(&block, cmd.capacity.capacity_reuse_warmup_from.as_deref())
        .map(Some)
        .map_err(|e| err(e.to_string()))
}
pub(super) fn warmup_count(auth: Option<&AuthorizedCapacitySessionBlock>, expected: u32) -> u32 {
    if auth.is_some_and(|a| !a.executes_warmup()) {
        0
    } else {
        expected
    }
}
pub(super) fn bind(
    evidence: &mut CapacityRunEvidence,
    auth: Option<AuthorizedCapacitySessionBlock>,
) {
    if let Some(auth) = auth {
        evidence.session = Some(auth.block().clone());
        evidence.warmup.acquisition = auth.acquisition().clone();
    }
}
