using Flux, Flux.CUDA

real_run = get(ENV, "CODESPEED_BRANCH", nothing) == "master"

# convenience macro to create a benchmark that requires synchronizing the GPU
macro async_benchmarkable(ex...)
    quote
        # use non-blocking sync to reduce overhead
        @benchmarkable CUDA.@sync blocking=false $(ex...)
    end
end

SUITE = BenchmarkGroup()
