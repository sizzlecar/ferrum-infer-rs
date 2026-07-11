use ferrum_interfaces::vnext::*;

fn use_after_commit<D: ResourceTransactionDriver>(
    reserved: ResourceTransaction<D, TransactionReserved>,
) {
    let _committed = reserved.commit();
    let _ = reserved.rollback();
}

fn main() {}
