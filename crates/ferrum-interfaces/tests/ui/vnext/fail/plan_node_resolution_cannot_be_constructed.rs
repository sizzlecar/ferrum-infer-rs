use ferrum_interfaces::vnext::*;
use std::collections::BTreeSet;

fn main() {
    let _ = PlanNodeResolution::new(
        NodeId::new("node.fake").unwrap(),
        Vec::new(),
        BTreeSet::new(),
        None,
    );
}
