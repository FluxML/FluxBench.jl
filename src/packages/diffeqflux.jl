function diffeqflux_add_neuralode(abstol = 1f-3, reltol = 1f-3, solver = Tsit5(), batchsize = 256, df_group = addgroup!(SUITE, "DiffEqFlux_NeuralODE"))
  down = Chain(flatten, Dense(784, 512, tanh))
  nn = Chain(Dense(512, 256, tanh),
             Dense(256, 256, tanh),
             Dense(256, 512, tanh))
  nn_ode = f -> NeuralODE(f, (0.f0, 1.f0), solver,
                          save_everystep = false,
                          reltol = reltol, abstol = abstol,
                          save_start = false)
  fc  = Chain(Dense(512, 10))

  function diffeqarray_to_array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
  end

  ip = rand(Float32, 784, batchsize)

  df_group["DiffEqFlux_Forward_Pass_NeuralODE_with_abstol_$(abstol)_reltol_$(reltol)_batchsize_$(batchsize)_and_solver_$(solver)"] = b = @benchmarkable(
    fw(model, gip),
    setup = (nn_gpu = gpu($nn);
             model = Chain($down,
			   $nn_ode(nn_gpu),
			   $diffeqarray_to_array,
                           $fc) |> gpu;
	     gip = gpu($ip)),
    teardown = (GC.gc(); CUDA.reclaim()))

  df_group["DiffEqFlux_Backward_Pass_NeuralODE_with_abstol_$(abstol)_reltol_$(reltol)_batchsize_$(batchsize)_and_solver_$(solver)"] = b = @benchmarkable(
    bw(model, gip),
    setup = (nn_gpu = $nn |> gpu;
             model = Chain($down,
                           $nn_ode(nn_gpu),
			   $diffeqarray_to_array,
			   $fc) |> gpu;
	     gip = gpu($ip)),
    teardown = (GC.gc(); CUDA.reclaim()))
end

function diffeqflux_add_neuralsde(batchsize = 16, ntrajectories = 100, df_group = addgroup!(SUITE, "DiffEqFlux_NeuralSDE"))
  diffusion = Chain(Dense(2, 8, tanh), Dense(8, 2))
  drift = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
  nn_sde = (f, g) -> NeuralDSDE(f, g, (0.0f0, 1.0f0), SOSRI(), abstol = 1f-1, reltol = 1f-1)

  function sdesol_to_array(x)
    xarr = gpu(x)
    return reshape(mean(reshape(xarr, size(xarr, 1), ntrajectories, size(xarr, 2)), dims = 2), size(xarr))
  end

  ip = repeat(rand(Float32, 2, batchsize), inner = (1, ntrajectories))

  df_group["DiffEqFlux_Forward_Pass_NeuralSDE_with_batchsize_$(batchsize)_and_ntrajectories_$(ntrajectories)"] = b = @benchmarkable(
    fw(model, gip),
    setup = (drift_gpu = gpu($drift);
             diffusion_gpu = gpu($diffusion);
	     model = Chain($nn_sde(drift_gpu, diffusion_gpu),
			   $sdesol_to_array);
	     gip = gpu($ip)),
    teardown = (GC.gc(); CUDA.reclaim()))

  df_group["DiffEqFlux_Backward_Pass_NeuralSDE_with_batchsize_$(batchsize)_and_ntrajectories_$(ntrajectories)"] = b = @benchmarkable(
    bw(model, gip),
    setup = (drift_gpu = gpu($drift);
             diffusion_gpu = $diffusion;
	     model = Chain($nn_sde(drift_gpu, diffusion_gpu),
			   $sdesol_to_array) |> gpu;
             gip = gpu($ip)),
    teardown = (GC.gc(); CUDA.reclaim()))
end

function diffeqflux_add_ffjord(ndims = 2, batchsize = 256, df_group = addgroup!(SUITE, "DiffEqFlux_FFJORD"))
  nn = Chain(Dense(ndims, ndims * 8, tanh), Dense(ndims * 8, ndims * 8, tanh), Dense(ndims * 8, ndims * 8, tanh), Dense(ndims * 8, ndims))
  cnf_ffjord = f -> FFJORD(f, (0.0f0, 1.0f0), Tsit5(), monte_carlo = true)
  ffjordsol_to_logpx(x) = -mean(x[1])[1]

  ip = rand(Float32, ndims, batchsize)
  e = randn(eltype(ip), size(ip))

  nsamples = batchsize
  function sample_from_learned_model(cnf_ffjord)
    pz = cnf_ffjord.basedist
    Z_samples = cu(rand(pz, nsamples))
    e = cu(randn(eltype(Z_samples), size(Z_samples)))
    ffjord_ = (u, p, t) -> DiffEqFlux.ffjord(u, p, t, cnf_ffjord.re, e; regularize = false, monte_carlo = false)
    _z = Zygote.@ignore similar(Z_samples, 1, size(Z_samples, 2))
    Zygote.@ignore fill!(_z, 0.0f0)
    prob = ODEProblem{false}(ffjord_, vcat(Z_samples, _z), (1.0, 0.0), cnf_ffjord.p)
    x_gen = solve(prob, cnf_ffjord.args...; sensealg = InterpolatingAdjoint(), cnf_ffjord.kwargs...)[1:end-1, :, end]
  end

  df_group["DiffEqFlux_Forward_Pass_FFJORD_with_batchsize_$(batchsize)_and_ndims_$(ndims)"] = b = @benchmarkable(
    CUDA.@sync ffjordsol_to_logpx(model(gip, model.p, e_gpu)),
    setup = (nn_gpu = gpu($nn);
             model = cnf_ffjord(nn_gpu);
             gip = gpu($ip);
             e_gpu = gpu($e)),
    teardown = (GC.gc(); CUDA.reclaim()))

  df_group["DiffEqFlux_Backward_Pass_FFJORD_with_batchsize_$(batchsize)_and_ndims_$(ndims)"] = b = @benchmarkable(
    CUDA.@sync gradient((m, x) -> sum(ffjordsol_to_logpx(m(x, m.p, e_gpu))), model, gip),
    setup = (nn_gpu = gpu($nn);
             model = cnf_ffjord(nn_gpu);
             gip = gpu($ip);
             e_gpu = gpu($e)),
    teardown = (GC.gc(); CUDA.reclaim()))

  df_group["DiffEqFlux_Sampling_FFJORD_with_nsamples_$(nsamples)_and_ndims_$(ndims)"] = b = @benchmarkable(
    fw(sample_from_learned_model, model),
    setup = (nn_gpu = gpu($nn);
             model = gpu($cnf_ffjord(nn_gpu))),
    teardown = (GC.gc(); CUDA.reclaim()))
end
