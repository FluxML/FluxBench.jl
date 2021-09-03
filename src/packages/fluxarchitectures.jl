function fluxarchitectures_add_darnn(encodersize, decodersize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures"))
    model = DARNN(inputsize, encodersize, decodersize, poollength, 1)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DARNN_Forward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_DARNN_Backward_Pass_encodersize_$(encodersize)_decodersize_$(decodersize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end
  
function fluxarchitectures_add_dsanet(locallength, nkernels, dmodel, hid, layers, nhead, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures"))
    model = DSANet(inputsize, poollength, locallength, nkernels, dmodel, hid, layers, nhead)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_DSANet_Forward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_DSANet_Backward_Pass_locallength_$(locallength)_nkernels_$(nkernels)_dmodel_$(dmodel)_hid_$(hid)_layers_$(layers)_nhead_$(nhead)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_lstnet(convlayersize, recurlayersize, poollength, skiplength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures"))
    model = LSTnet(inputsize, convlayersize, recurlayersize, poollength, skiplength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_LSTNet_Forward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_LSTNet_Backward_Pass_convlayersize_$(convlayersize)_recurlayersize_$(recurlayersize)_skiplength_$(skiplength)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

function fluxarchitectures_add_tpalstm(hiddensize, poollength, inputsize, datalength, fa_group=addgroup!(SUITE, "FluxArchitectures"))
    model = TPALSTM(inputsize, hiddensize, poollength)
    ip = randn(Float32, inputsize, poollength, 1, datalength)

    fa_group["FluxArchitectures_TPALSTM_Forward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        fw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    fa_group["FluxArchitectures_TPALSTM_Backward_Pass_hiddensize_$(hiddensize)_poollength_$(poollength)_inputsize_$(inputsize)_datalength_$(datalength)"] = @benchmarkable(
        bw(model, gip), setup = (model = gpu($model); gip = gpu($ip)),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end