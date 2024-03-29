struct FA_GPU end
struct FA_CPU end


# GPU Tests

function fluxarchitectures_add_darnn(::FA_GPU, encodersize, decodersize, poollength, 
                                    inputsize, datalength, 
                                    fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["DARNN_forw_encoder_$(encodersize)_decoder_$(decodersize)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["DARNN_back_encoder_$(encodersize)_decoder_$(decodersize)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_dsanet(::FA_GPU, locallength, nkernels, dmodel, hid, 
                                      layers, nhead, poollength, inputsize, datalength, 
                                      fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DSANet(inputsize, poollength, locallength, nkernels, dmodel, hid, layers, nhead)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["DSANet_forw_lay_$(layers)_nhead_$(nhead)_pool_$(poollength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["DSANet_back_lay_$(layers)_nhead_$(nhead)_pool_$(poollength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_lstnet(::FA_GPU, convlayersize, recurlayersize, poollength, 
                                      skiplength, inputsize, datalength, 
                                      fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["LSTNet_forw_conv_$(convlayersize)_recur_$(recurlayersize)_skip_$(skiplength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["LSTNet_back_conv_$(convlayersize)_recur_$(recurlayersize)_skip_$(skiplength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_tpalstm(::FA_GPU, hiddensize, poollength, inputsize,
                                       datalength,
                                       fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = TPALSTM(inputsize, hiddensize, poollength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["TPALSTM_forw_hidden_$(hiddensize)_poollength_$(poollength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["TPALSTM_back_hidden_$(hiddensize)_poollength_$(poollength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end


# CPU Tests

function fluxarchitectures_add_darnn(::FA_CPU, encodersize, decodersize, poollength,
                                     inputsize, datalength, 
                                     fa_group=addgroup!(SUITE, "FluxArchitectures_CPU"))
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["DARNN_forw_encoder_$(encodersize)_decoder_$(decodersize)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["DARNN_back_encoder_$(encodersize)_decoder_$(decodersize)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end

function fluxarchitectures_add_dsanet(::FA_CPU, locallength, nkernels, dmodel, hid, 
                                      layers, nhead, poollength, inputsize, datalength,
                                      fa_group=addgroup!(SUITE, "FluxArchitectures_CPU"))
    model = DSANet(inputsize, poollength, locallength, nkernels, dmodel, hid, layers, nhead)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["DSANet_forw_lay_$(layers)_nhead_$(nhead)_pool_$(poollength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["DSANet_back_lay_$(layers)_nhead_$(nhead)_pool_$(poollength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end

function fluxarchitectures_add_lstnet(::FA_CPU, convlayersize, recurlayersize, poollength, 
                                      skiplength, inputsize, datalength, 
                                      fa_group=addgroup!(SUITE, "FluxArchitectures_CPU"))
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["LSTNet_forw_conv_$(convlayersize)_recur_$(recurlayersize)_skip_$(skiplength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["LSTNet_back_conv_$(convlayersize)_recur_$(recurlayersize)_skip_$(skiplength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end

function fluxarchitectures_add_tpalstm(::FA_CPU, hiddensize, poollength, inputsize,
                                       datalength, 
                                       fa_group=addgroup!(SUITE, "FluxArchitectures_CPU"))
    model = TPALSTM(inputsize, hiddensize, poollength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["TPALSTM_forw_hidden_$(hiddensize)_poollength_$(poollength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["TPALSTM_back_hidden_$(hiddensize)_poollength_$(poollength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end