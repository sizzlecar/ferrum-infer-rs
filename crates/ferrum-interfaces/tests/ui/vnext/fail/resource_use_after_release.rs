use ferrum_interfaces::vnext::*;

fn use_after_release<D: ResourceTransactionDriver>(
    committed: ResourceTransaction<D, TransactionCommitted>,
) {
    let _released = committed.release();
    let _ = committed.lease();
}

fn main() {}
