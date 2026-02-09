use ferrum_interfaces::tensor::utils::*;

#[test]
fn matmul_output_shape_valid_and_invalid() {
    let out = matmul_output_shape(&[2, 3], &[3, 4]).unwrap();
    assert_eq!(out, vec![2, 4]);
    let err = matmul_output_shape(&[2, 3], &[2, 4]).err().unwrap();
    let msg = format!("{}", err);
    assert!(msg.to_lowercase().contains("mismatch"));
}

#[test]
fn broadcast_helpers() {
    assert!(are_broadcastable(&[2, 3], &[1, 3]));
    assert!(!are_broadcastable(&[2, 3], &[2, 2]));
    let out = broadcast_shapes(&[2, 1, 3], &[1, 4, 3]).unwrap();
    assert_eq!(out, vec![2, 4, 3]);
}
