using BenchmarkTools, TimerOutputs
using Flux.CUDA
# using Torch - If we want to compare progress

# CUDA.device!(3)

function fw(m, ip)
    CUDA.@sync m(ip)
end

function benchmark_cu(io, batchsize = 64)
  resnet = ResNet()
  ip = rand(Float32, 224, 224, 3, batchsize)

  # gresnet = resnet |> gpu
  # gip = gpu(ip)

  b = @benchmarkable(
        fw(gresnet, gip),
        setup=(gresnet = $resnet |> gpu;
               gip = gpu($ip)),
        teardown=(GC.gc(); CUDA.reclaim()))

  write(io, run(b))
  write(io, "\n\n")
end

function bw(m, ip)
  gs = CUDA.@sync gradient((m, x) -> sum(m(x)), m, ip)
end

function benchmark_bw_cu(io, batchsize = 64)
  resnet = ResNet()
  ip = rand(Float32, 224, 224, 3, batchsize)

  # gresnet = resnet |> gpu
  # gip = gpu(ip)

  b = @benchmarkable(
        bw(gresnet, gip),
        setup=(gresnet = $resnet |> gpu;
   	       gip = gpu($ip)),
        teardown=(GC.gc(); CUDA.reclaim()))

  write(io, run(b))
  write()
end

function bench()
  for n in (5, 15, 32, 64, 96)
    filename = "bench.txt"
    to = TimerOutput()
    open(filename, "w") do io
      benchmark_bw_cu(io, n)
    end
  
    open(filename, "w") do io
      benchmark_cu(io, n)
    end
  end
end
