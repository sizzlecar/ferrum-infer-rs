#[test]
fn vnext_compile() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/vnext/pass/*.rs");
    tests.compile_fail("tests/ui/vnext/fail/*.rs");
}
