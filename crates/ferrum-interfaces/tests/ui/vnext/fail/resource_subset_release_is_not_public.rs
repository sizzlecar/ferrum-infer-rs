use ferrum_interfaces::vnext::*;

fn release_one<D: ResourceTransactionDriver>(
    transaction: &mut ResourceTransaction<D, TransactionCommitted>,
    resource_id: ResourceId,
) {
    let _ = transaction.release_subset(&[resource_id]);
}

fn main() {}
