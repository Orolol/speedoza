# Benches

Benchmark harnesses live here once the CUDA backend is linked.

Planned:

- Layout discovery throughput on the real `model.safetensors`.
- NVFP4 GEMM microbenchmarks vs cuBLASLt heuristics.
- DeltaNet recurrent decode kernel.
- TurboQuant fused QK/V attention path.
- End-to-end decode against vLLM on identical prompts/context lengths.

