#pragma once

#include <cuda_runtime.h>

// Defined in runtime.cu. Returns the ambient CUDA stream every kernel in this
// library should launch on. Defaults to the legacy default stream (0); the
// host can swap to a named stream (e.g. via qwen36_set_active_stream) so the
// engine can capture the decode forward into a CUDA graph.
extern "C" cudaStream_t qwen36_internal_active_stream();

// Returns the secondary "prefetch" stream the engine has registered (or
// nullptr if none). Used by productive-spin and megakernel paths to dispatch
// concurrent prefetch / helper kernels alongside the main stream. The two
// streams synchronize via qwen36_cuda_event_record / _stream_wait_event so
// the whole sequence remains graph-captureable.
extern "C" cudaStream_t qwen36_internal_prefetch_stream();
