use ferrum_interfaces::vnext::{
    ResourceTransaction, ResourceTransactionDriver, TransactionCommitted,
};

fn handoff_then_release<D>(transaction: ResourceTransaction<D, TransactionCommitted>)
where
    D: ResourceTransactionDriver + 'static,
{
    let _resources = transaction.into_plan_runtime();
    let _released = transaction.release();
}

fn main() {}
