# Security

This project loads local model files, mmaps safetensors, and calls CUDA kernels through a native shared library. Treat model directories and shared libraries as trusted inputs.

## Supported Scope

- Local CLI/library use.
- No HTTP server or multi-tenant serving layer.
- No sandbox for untrusted model checkpoints.

## Reporting

Open a private issue or contact the repository owner if this repo is published under an organization. Include:

- Commit hash.
- Host OS, NVIDIA driver, CUDA version, and GPU.
- Reproducer command.
- Whether the issue requires a specific model file.

