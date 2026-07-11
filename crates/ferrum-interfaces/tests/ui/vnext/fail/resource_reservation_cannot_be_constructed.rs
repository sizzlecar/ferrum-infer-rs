use ferrum_interfaces::vnext::*;

fn main() {
    let _ = ResourceReservation::new(
        ResourceId::new("resource/forged").unwrap(),
        RequestIdentity::new("request/forged").unwrap(),
        None,
        64,
        16,
        BufferUsage::State,
        ElementType::U8,
        1,
    );
}
