module FluxBench

using Flux, Metalhead
using BenchmarkTools, TimerOutputs

const MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)

include("benchmarkutils.jl")
include("bench.jl")

end # module
