function diffeqflux_add_neuralode(
    abstol=1.0f-3, reltol=1.0f-3, solver=Tsit5(), batchsize=256, df_group=addgroup!(SUITE, "DiffEqFlux_NeuralODE")
)
    down = Chain(Flux.flatten, Dense(784, 512, tanh))
    nn = Chain(Dense(512, 256, tanh), Dense(256, 256, tanh), Dense(256, 512, tanh))
    nn_ode =
        f -> NeuralODE(f, (0.0f0, 1.0f0), solver; save_everystep=false, reltol=reltol, abstol=abstol, save_start=false)
    fc = Chain(Dense(512, 10))

    function diffeqarray_to_array(x)
        xarr = gpu(x)
        return reshape(xarr, size(xarr)[1:2])
    end

    ip = rand(Float32, 784, batchsize)

    df_group["DiffEqFlux_FWD_NODE_atol_$(abstol)_rtol_$(reltol)_bsz_$(batchsize)_$(solver)"] = @benchmarkable(
        fw(model, gip),
        setup = (nn_gpu = gpu($nn);
        model = gpu(Chain($down, $nn_ode(nn_gpu), $diffeqarray_to_array, $fc));
        gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    df_group["DiffEqFlux_BWD_NODE_atol_$(abstol)_rtol_$(reltol)_bsz_$(batchsize)_$(solver)"] = @benchmarkable(
        bw(model, gip),
        setup = (nn_gpu = gpu($nn);
        model = gpu(Chain($down, $nn_ode(nn_gpu), $diffeqarray_to_array, $fc));
        gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    return nothing
end

function diffeqflux_add_neuralsde(batchsize=16, ntrajectories=100, df_group=addgroup!(SUITE, "DiffEqFlux_NeuralSDE"))
    diffusion = Chain(Dense(2, 8, tanh), Dense(8, 2))
    drift = Chain(Dense(2, 32, tanh), Dense(32, 32, tanh), Dense(32, 2))
    nn_sde = (f, g) -> NeuralDSDE(f, g, (0.0f0, 1.0f0), SOSRI(); abstol=1.0f-1, reltol=1.0f-1)

    sdesol_to_array(x) = mean(gpu(x); dims=2)

    ip = repeat(rand(Float32, 2, batchsize); inner=(1, ntrajectories))

    df_group["DiffEqFlux_FWD_NSDE_bsize_$(batchsize)_ntraj_$(ntrajectories)"] = @benchmarkable(
        fw(model, gip),
        setup = (drift_gpu = gpu($drift);
        diffusion_gpu = gpu($diffusion);
        model = Chain($nn_sde(drift_gpu, diffusion_gpu), $sdesol_to_array);
        gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    df_group["DiffEqFlux_BWD_NSDE_bsize_$(batchsize)_ntraj_$(ntrajectories)"] = @benchmarkable(
        bw(model, gip),
        setup = (drift_gpu = gpu($drift);
        diffusion_gpu = gpu($diffusion);
        model = gpu(Chain($nn_sde(drift_gpu, diffusion_gpu), $sdesol_to_array));
        gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    return nothing
end

function diffeqflux_add_ffjord(ndims=2, batchsize=256, df_group=addgroup!(SUITE, "DiffEqFlux_FFJORD"))
    nn = Chain(
        Dense(ndims, ndims * 8, tanh),
        Dense(ndims * 8, ndims * 8, tanh),
        Dense(ndims * 8, ndims * 8, tanh),
        Dense(ndims * 8, ndims),
    )
    function cnf_ffjord(f, e)
        ffjord = FFJORD(f, (0.0f0, 1.0f0), Tsit5(); monte_carlo=true)
        return x -> first(ffjord(x, ffjord.p, e))
    end

    function ffjord_sampling(f, e)
        model = FFJORD(f, (0.0f0, 1.0f0), Tsit5(); monte_carlo=true)
        return _ -> DiffEqFlux.backward_ffjord(model, batchsize, model.p, e)
    end

    ip = rand(Float32, ndims, batchsize)
    e = randn(eltype(ip), size(ip))

    df_group["DiffEqFlux_FWD_FFJORD_with_bsz_$(batchsize)_ndims_$(ndims)"] = @benchmarkable(
        fw(model, gip), setup = (nn_gpu = gpu($nn);
        e_gpu = gpu($e);
        model = cnf_ffjord($nn_gpu, $e_gpu);
        gip = gpu($ip)), teardown = (GC.gc(); CUDA.reclaim())
    )

    df_group["DiffEqFlux_BWD_FFJORD_with_bsz_$(batchsize)_ndims_$(ndims)"] = @benchmarkable(
        bw(model, gip), setup = (nn_gpu = gpu($nn);
        e_gpu = gpu($e);
        model = cnf_ffjord($nn_gpu, $e_gpu);
        gip = gpu($ip)), teardown = (GC.gc(); CUDA.reclaim())
    )

    df_group["DiffEqFlux_Samp_FFJORD_with_bsz_$(batchsize)_ndims_$(ndims)"] = @benchmarkable(
        fw(model, gip),
        setup = (nn_gpu = gpu($nn);
        e_gpu = gpu($e);
        model = ffjord_sampling($nn_gpu, $e_gpu);
        gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    return nothing
end
