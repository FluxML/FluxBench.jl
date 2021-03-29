module FluxBench

using Flux, Metalhead, ObjectDetector
using BenchmarkTools, TimerOutputs
using HTTP, JSON, FileIO
using Flux.CUDA
# using Torch - If we want to compare progress

const MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)

SUITE = BenchmarkGroup()

include("benchmarkutils.jl")
include("packages/objectdetector.jl")
include("bench.jl")

results = run(SUITE, verbose = true)
println(results)

flat_results = []
flatten(results)

HTTP.post("$(ENV["CODESPEED_SERVER"])/result/add/json/",
          ["Content-Type" => "application/x-www-form-urlencoded"],
          HTTP.URIs.escapeuri(Dict("json" => JSON.json(flat_results))))

end # module
