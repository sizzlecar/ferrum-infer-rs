use ferrum_interfaces::vnext::{ResourceTransaction, ResourceTransactionDriver, TransactionNew};

fn duplicate<D: ResourceTransactionDriver>(transaction: ResourceTransaction<D, TransactionNew>) {
    let _duplicate = transaction.clone();
}

fn main() {}
