use ferrum_runtime::ResolvedValueBinding;

fn first_component(binding: &ResolvedValueBinding) {
    let _ = binding.resource_id();
}

fn main() {}
