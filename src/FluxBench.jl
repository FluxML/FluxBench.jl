module FluxBench

using Flux, Metalhead
using BenchmarkTools, TimerOutputs

const MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)

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
