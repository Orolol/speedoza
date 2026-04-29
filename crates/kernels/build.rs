use std::env;

fn main() {
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=QWEN36_FP4_KERNEL_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");

    if let Some(dir) = env::var_os("QWEN36_FP4_KERNEL_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", dir.to_string_lossy());
    } else {
        println!("cargo:rustc-link-search=native=target/cuda");
    }
    println!("cargo:rustc-link-lib=dylib=qwen36_fp4_kernels");
}

