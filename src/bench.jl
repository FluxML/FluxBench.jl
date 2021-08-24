# CUDA.device!(3)

group = addgroup!(SUITE, "Metalhead")

function benchmark_cu(model, batchsize = 64)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  group["Forward_Pass_$(model)_with_batchsize_$(batchsize)"] = b = @benchmarkable(
        fw(gresnet, gip),
        setup = (gresnet = $resnet |> gpu;
               gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim()))
end

function benchmark_bw_cu(model, batchsize = 64)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  group["Backwards_Pass_$(model)_with_batchsize_$(batchsize)"] = b = @benchmarkable(
        bw(gresnet, gip),
        setup = (gresnet = $resnet |> gpu;
   	       gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim()))

end

function bench()
  for model in MODELS, n in (5, 15)
    # we can go higher with the batchsize
    # but the CI machines would have variable VRAM
    # so be conservative
    # TODO: add larger batchsize for full benchmarking runs
    benchmark_bw_cu(model(), n)
    benchmark_cu(model(), n)
  end

  # ObjectDetector
  for model in [ObjectDetector.YOLO.v3_608_COCO, ObjectDetector.YOLO.v3_tiny_416_COCO], batchsize in [1, 3]
    objectdetector_add_yolo_fw(model, batchsize)
  end

  # # DiffEqFlux
  # ## NeuralODE
  # for tol in (1f-3, 1f-5, 1f-8), b in (4, 16, 64, 256)
  #   diffeqflux_add_neuralode(tol, tol, tol > 1f-8 ? Tsit5() : Vern7(), b)
  # end
  # ## NeuralSDE
  # for b in (4, 16, 64), traj in (1, 10, 32)
  #   diffeqflux_add_neuralsde(b, traj)
  # end
  # ## FFJORD
  # for b in (4, 16, 64, 256), ndims in (2, 4, 8)
  #   diffeqflux_add_ffjord(b, ndims)
  # end
end
