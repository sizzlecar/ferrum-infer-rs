use ferrum_runtime::ResourceCommitView;

fn escape<'a>(view: &'a ResourceCommitView<'a, Vec<u8>>) -> &'static Vec<u8> {
    view.buffer()
}

fn main() {}
