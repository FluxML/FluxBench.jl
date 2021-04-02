group = addgroup!(SUITE, "DiffEqFlux")

function diffeqflux_add_neuralode(abstol = 1f-3, reltol = 1f-3, solver = Tsit5(), batchsize = 256)
  down = Chain(flatten, Dense(784, 512, tanh))
  nn = Chain(Dense(512, 256, tanh),
             Dense(256, 256, tanh),
             Dense(256, 512, tanh))
  nn_ode = f -> NeuralODE(f, (0.f0, 1.f0), solver,
                          save_everystep = false,
                          reltol = reltol, abstol = abstol,
                          save_start = false)
  fc  = Chain(Dense(512, 10))

  function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return reshape(xarr, size(xarr)[1:2])
  end

  ip = rand(Float32, 784, batchsize)

  group["DiffEqFlux - Forward Pass - NeuralODE with abstol $abstol, reltol $reltol, batchsize $batchsize, and solver $solver"] = b = @benchmarkable(
    fw(model, gip)
    setup = (nn_gpu = $nn |> gpu; model = Chain($down, $nn_ode(nn_gpu), $DiffEqArray_to_Array, $fc); gip = $ip |> gpu)
    teardown = (GC.gc(); CUDA.reclaim()))

  group["DiffEqFlux - Backward Pass - NeuralODE with abstol $abstol, reltol $reltol, batchsize $batchsize, and solver $solver"] = b = @benchmarkable(
    bw(model, gip)
    setup = (nn_gpu = $nn |> gpu; model = Chain($down, $nn_ode(nn_gpu), $DiffEqArray_to_Array, $fc); gip = $ip |> gpu)
    teardown = (GC.gc(); CUDA.reclaim()))
end