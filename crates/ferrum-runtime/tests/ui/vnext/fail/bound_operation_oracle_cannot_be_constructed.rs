use ferrum_runtime::BoundOperationOracle;

fn forge<'a>() -> BoundOperationOracle<'a> {
    BoundOperationOracle {
        requested_operation: panic!(),
        terminal_operation: panic!(),
        registered: panic!(),
    }
}

fn main() {
    let _ = forge();
}
