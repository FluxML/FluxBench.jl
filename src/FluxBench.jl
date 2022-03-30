module FluxBench

using Flux, Metalhead, ObjectDetector
using Flux3D
using FluxArchitectures

# using DiffEqFlux
# using OrdinaryDiffEq, StochasticDiffEq, Distributions
using BenchmarkTools, TimerOutputs
using HTTP, JSON, FileIO
using Flux.CUDA
using Statistics
using Zygote
# using Torch - If we want to compare progress

SUITE = BenchmarkGroup()

include("utils.jl")
include("packages/objectdetector.jl")
include("packages/transformers.jl")
include("packages/flux3d.jl")
# include("packages/fluxarchitectures.jl")
include("packages/diffeqflux.jl")
include("bench.jl")


function submit(submit = false, SUITE = SUITE)
  bench()
  warmup(SUITE, verbose = true)
  results = run(SUITE, verbose = true)
  println(results)
  
  flat_results = flatten(results)

  print(JSON.json(flat_results))
  println()

  if submit
    HTTP.post("$(ENV["CODESPEED_SERVER"])/result/add/json/",
              ["Content-Type" => "application/x-www-form-urlencoded"],
              HTTP.URIs.escapeuri(Dict("json" => JSON.json(flat_results))))
  end
end

end # module
