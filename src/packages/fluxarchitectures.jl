struct FA_GPU end
struct FA_CPU end


# GPU Tests

function fluxarchitectures_add_darnn(::FA_GPU, encodersize, decodersize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DARNN_GPU_Forward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_DARNN_GPU_Backward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
)
end
  
function fluxarchitectures_add_dsanet(::FA_GPU, locallength, nkernels, dmodel, hid, layers, nhead, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DSANet(inputsize, poollength, locallength, nkernels, dmodel, hid, layers, nhead)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DSANet_GPU_Forward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_DSANet_GPU_Backward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_lstnet(::FA_GPU, convlayersize, recurlayersize, poollength, skiplength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_LSTNet_GPU_Forward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_LSTNet_GPU_Backward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_tpalstm(::FA_GPU, hiddensize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = TPALSTM(inputsize, hiddensize, poollength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_TPALSTM_GPU_Forward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_TPALSTM_GPU_Backward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end


# CPU Tests

function fluxarchitectures_add_darnn(::FA_CPU, encodersize, decodersize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DARNN_CPU_Forward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["FluxArchitectures_DARNN_CPU_Backward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
)
end
  
function fluxarchitectures_add_dsanet(::FA_CPU, locallength, nkernels, dmodel, hid, layers, nhead, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = DSANet(inputsize, poollength, locallength, nkernels, dmodel, hid, layers, nhead)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DSANet_CPU_Forward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["FluxArchitectures_DSANet_CPU_Backward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end

function fluxarchitectures_add_lstnet(::FA_CPU, convlayersize, recurlayersize, poollength, skiplength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_LSTNet_CPU_Forward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["FluxArchitectures_LSTNet_CPU_Backward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end

function fluxarchitectures_add_tpalstm(::FA_CPU, hiddensize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures_GPU"))
    model = TPALSTM(inputsize, hiddensize, poollength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_TPALSTM_CPU_Forward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )

    fa_group["FluxArchitectures_TPALSTM_CPU_Backward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw_cpu(model, gip), setup = (model = $model; gip = $ip),
        teardown = ()
    )
end