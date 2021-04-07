# CUDA.device!(3)

group = addgroup!(SUITE, "Metalhead")

function benchmark_cu(io, model, batchsize = 64)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  # gresnet = resnet |> gpu
  # gip = gpu(ip)

  group["Forward Pass - $model with batchsize $batchsize"] = b = @benchmarkable(
        fw(gresnet, gip),
        setup=(gresnet = $resnet |> gpu;
               gip = gpu($ip)),
        teardown=(GC.gc(); CUDA.reclaim()))

  # write(io, run(b))
  # write(io, "\n\n")
end

function benchmark_bw_cu(io, model, batchsize = 64)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  # gresnet = resnet |> gpu
  # gip = gpu(ip)

  group["Backwards Pass - $model with batchsize $batchsize"] = b = @benchmarkable(
        bw(gresnet, gip),
        setup=(gresnet = $resnet |> gpu;
   	       gip = gpu($ip)),
        teardown=(GC.gc(); CUDA.reclaim()))

  # write(io, run(b))
  # write()
end

function bench()
  for model in MODELS, n in (5, 15, 32, 64, 96)
    filename = "bench.txt"
    to = TimerOutput()
    # open(filename, "w") do io
      benchmark_bw_cu(io, model(), n)
    # end

    # open(filename, "w") do io
      benchmark_cu(io, model(), n)
    # end
  end

  # ObjectDetector
  for model in [ObjectDetector.YOLO.v3_608_COCO, ObjectDetector.v3_tiny_416_COCO], batchsize in [1, 3]
    objectdetector_add_yolo_fw(model = model, batchsize = batchsize)
  end

  # DiffEqFlux
  ## NeuralODE
  for tol in (1f-3, 1f-5, 1f-8), b in (4, 16, 64, 256)
    diffeqflux_add_neuralode(tol, tol, tol > 1f-8 ? Tsit5() : Vern7(), b)
  end
  ## NeuralSDE
  for b in (4, 16, 64), traj in (1, 10, 32)
    diffeqflux_add_neuralsde(b, traj)
  end
  ## FFJORD
  for b in (4, 16, 64, 256), ndims in (2, 4, 8)
    diffeqflux_add_ffjord(b, ndims)
  end
end
