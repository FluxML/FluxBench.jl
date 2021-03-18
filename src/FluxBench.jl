module FluxBench

using Flux, Metalhead
using BenchmarkTools, TimerOutputs

MODELS = (ResNet, DenseNet, GoogleNet, VGG19, SqueezeNet)
include("bench.jl")

end # module
