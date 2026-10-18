use ferrum_runtime::OperationDispatch;

fn raw_command_escape_is_absent() {
    let _ = OperationDispatch::encode;
}

fn main() {}
