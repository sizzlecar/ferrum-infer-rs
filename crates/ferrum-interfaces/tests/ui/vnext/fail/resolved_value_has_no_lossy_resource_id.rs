use ferrum_interfaces::vnext::ResolvedValueBinding;

fn first_component(binding: &ResolvedValueBinding) {
    let _ = binding.resource_id();
}

fn main() {}
