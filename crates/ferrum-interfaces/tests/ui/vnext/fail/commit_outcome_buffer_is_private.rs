use ferrum_interfaces::vnext::ResourceCommitOutcome;

fn raw_buffer<B>(outcome: &ResourceCommitOutcome<B>) {
    let _ = outcome.buffer();
}

fn main() {}
