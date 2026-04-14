fn main() {
    // Link Apple Accelerate framework for cblas_sgemm
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
