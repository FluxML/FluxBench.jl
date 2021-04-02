using Flux, Flux.CUDA

const REAL_RUN = get(ENV, "CODESPEED_BRANCH", nothing) == "master"

# convenience macro to create a benchmark that requires synchronizing the GPU
macro async_benchmarkable(ex...)
    quote
        # use non-blocking sync to reduce overhead
        @benchmarkable CUDA.@sync blocking = false $(ex...)
    end
end

basedata = Dict(
        "branch"        => ENV["CODESPEED_BRANCH"],
        "commitid"      => ENV["CODESPEED_COMMIT"],
        "project"       => ENV["CODESPEED_PROJECT"],
        "environment"   => ENV["CODESPEED_ENVIRONMENT"],
        "executable"    => ENV["CODESPEED_EXECUTABLE"]
)

# convert nested groups of benchmark to flat dictionaries of results
function flatten(results, prefix = "")
  for (key,value) in results
    if value isa BenchmarkGroup
      flatten(value, "$prefix$key/")
    else
      @assert value isa BenchmarkTools.Trial

      # codespeed reports maxima, but those are often very noisy.
      # get rid of measurements that unnecessarily skew the distribution.
      rmskew!(value)

      push!(flat_results,
        Dict(basedata...,
          "benchmark" => "$prefix$key",
          "result_value" => median(value).time / 1e9,
          "min" => minimum(value).time / 1e9,
          "max" => maximum(value).time / 1e9))
    end
  end
end

# Do a forward pass
function fw(m, ip)
    CUDA.@sync m(ip)
end

# Do a forward + backward pass
function bw(m, ip)
  gs = CUDA.@sync gradient((m, x) -> sum(m(x)), m, ip)
end