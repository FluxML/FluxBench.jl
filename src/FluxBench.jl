module FluxBench

using Flux, Metalhead# , ObjectDetector, DiffEqFlux
# using OrdinaryDiffEq, StochasticDiffEq, Distributions
using BenchmarkTools, TimerOutputs
using HTTP, JSON, FileIO
using Flux.CUDA
using Statistics
using Zygote
# using Torch - If we want to compare progress

const MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)

SUITE = BenchmarkGroup()

include("benchmarkutils.jl")
# include("packages/objectdetector.jl")
# include("packages/diffeqflux.jl")
include("bench.jl")


function submit(SUITE = SUITE)
  bench()
  warmup(SUITE, verbose = true)
  results = run(SUITE, verbose = true)
  println(results)
  
  flat_results = flatten(results)

  print(JSON.json(flat_results))
  println()
  HTTP.post("$(ENV["SECRET_CODESPEED_SERVER"])/result/add/json/",
            ["Content-Type" => "application/x-www-form-urlencoded"],
            HTTP.URIs.escapeuri(Dict("json" => JSON.json(flat_results))))
end

end # module
