using Transformers

function transformer_add_trf(::Type{Transformer}, layern, batchn,
                                     trf_group = addgroup!(SUITE, "Transformers"))
    hidden, head, intermediate = 512, 8, 2048
    encoder = Chain((Transformer(hidden, head, intermediate; act = gelu) for i = 1:layern)...)
    ip = randn(512, 100, batchn)

    trf_group["Transformers_Forward_Pass_nlayer_$(layern)_nbatch_$(batchn)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($encoder); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    trf_group["Transformers_Backward_Pass_nlayer_$(layern)_nbatch_$(batchn)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($encoder); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

end
