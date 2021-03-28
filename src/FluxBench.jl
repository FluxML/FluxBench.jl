module FluxBench

using Flux, Metalhead
using BenchmarkTools, TimerOutputs
using HTTP, JSON

const MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)

SUITE = BenchmarkGroup()

include("benchmarkutils.jl")
include("bench.jl")

results = run(SUITE, verbose = true)
println(results)

flat_results = []
flatten(results)

HTTP.post("$(ENV["CODESPEED_SERVER"])/result/add/json/",
          ["Content-Type" => "application/x-www-form-urlencoded"],
          HTTP.URIs.escapeuri(Dict("json" => JSON.json(flat_results))))

end # module
