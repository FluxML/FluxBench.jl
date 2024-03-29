# CUDA.device!(3)

const METALHEAD_MODELS = ((ResNet18, 18), (ResNet34, 34), (ResNet50, 50),
                          (() -> DenseNet((6, 12, 24, 16)), 121),
                          (GoogLeNet, nothing),
                          (VGG19, 19)) #, SqueezeNet)

function benchmark_cu(model, batchsize = 64, config = nothing, group = nothing)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  group["Forw_$(typeof(model))_$(config)_batch_$(batchsize)"] = b = @benchmarkable(
        fw(gresnet, gip),
        setup = (gresnet = $resnet |> gpu;
               gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim()))
end

function benchmark_bw_cu(model, batchsize = 64, config = nothing, group = nothing)
  resnet = model
  ip = rand(Float32, 224, 224, 3, batchsize)

  group["Back_$(typeof(model))_$(config)_batch_$(batchsize)"] = b = @benchmarkable(
        bw(gresnet, gip),
        setup = (gresnet = $resnet |> gpu;
   	       gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim()))

end

function bench()
  mhead_group = addgroup!(SUITE, "Metalhead")
  for model in METALHEAD_MODELS, n in (5, 10)
    # we can go higher with the batchsize
    # but the CI machines would have variable VRAM
    # so be conservative
    # TODO: add larger batchsize for full benchmarking runs
    m, config = model
    benchmark_bw_cu(m(), n, config, mhead_group)
    benchmark_cu(m(), n, config, mhead_group)
  end

  # ObjectDetector
  od_group = addgroup!(SUITE, "ObjectDetector")
  for model in [ObjectDetector.YOLO.v3_608_COCO, ObjectDetector.YOLO.v3_tiny_416_COCO], batchsize in [1, 3]
    objectdetector_add_yolo_fw(model, batchsize, od_group)
  end

  # # DiffEqFlux
  # ## NeuralODE
  # df_group = addgroup!(SUITE, "DiffEqFlux_NeuralODE")
  # for tol in (1f-3, 1f-5, 1f-8), b in (4, 16, 64, 256)
  #   diffeqflux_add_neuralode(tol, tol, tol > 1f-8 ? Tsit5() : Vern7(), b, df_group)
  # end
  # ## NeuralSDE
  # df_group = addgroup!(SUITE, "DiffEqFlux_NeuralSDE")
  # for b in (4, 16, 64), traj in (1, 10, 32)
  #   diffeqflux_add_neuralsde(b, traj, df_group)
  # end
  # ## FFJORD
  # df_group = addgroup!(SUITE, "DiffEqFlux_FFJORD")
  # for b in (4, 16, 64, 256), ndims in (2, 4, 8)
  #   diffeqflux_add_ffjord(b, ndims, df_group)
  # end

  # Transformers
  trf_group = addgroup!(SUITE, "Transformers")
  transformer_add_trf(Transformer, 12, 32, trf_group)
  transformer_add_trf(Bert, 8, trf_group)


  # Flux3D.jl
  flux3d_group = addgroup!(SUITE, "Flux3D")
  flux3d_add_trimesh(flux3d_group)


  # FluxArchitectures
  fa_gpu_group = addgroup!(SUITE, "FluxArchitectures_GPU")
  fluxarchitectures_add_darnn(FA_GPU(), 5, 5, 10, 10, 300, fa_gpu_group)
  fluxarchitectures_add_dsanet(FA_GPU(), 3, 3, 4, 1, 3, 2, 10, 50, 1000, fa_gpu_group)
  fluxarchitectures_add_lstnet(FA_GPU(), 2, 3, 10, 60, 20, 500, fa_gpu_group)
  fluxarchitectures_add_tpalstm(FA_GPU(), 10, 10, 10, 300, fa_gpu_group)
  fa_cpu_group = addgroup!(SUITE, "FluxArchitectures_CPU")
  fluxarchitectures_add_darnn(FA_CPU(), 5, 5, 10, 10, 300, fa_cpu_group)
  fluxarchitectures_add_dsanet(FA_CPU(), 3, 3, 4, 1, 3, 2, 10, 50, 1000, fa_cpu_group)
  fluxarchitectures_add_lstnet(FA_CPU(), 2, 3, 10, 60, 20, 500, fa_cpu_group)
  fluxarchitectures_add_tpalstm(FA_CPU(), 10, 10, 10, 300, fa_cpu_group)

  # GeometricFlux
  geoflux_group = addgroup!(SUITE, "GeometricFlux")
  for hidden_dim in [16, 32], bch_sz in [8, 16]
    geometricflux_add_gcn(geoflux_group, GCN, hidden_dim, bch_sz)
  end
end
