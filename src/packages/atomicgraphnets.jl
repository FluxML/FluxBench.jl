using Pkg
using Pkg.Artifacts
using Serialization

flux_versions = ["0.11", "0.12"]
env_dirs = Dict(k=>joinpath(@__DIR__, "AtomicGraphNets_deps", "v$k") for k in flux_versions)

function agn_setup(flux_version, batchsize, num_conv, hidden_layer_width, num_hidden_layers)
    # sanity checks
    @assert flux_version in flux_versions
    @assert batchsize <= 100

    cd(env_dirs[flux_version])
    Pkg.activate(".")
    
    @eval using AtomicGraphNets

    # load up the (pre-shuffled) data
    artpaths = Dict("0.11"=>artifact"v011_data", "0.12"=>artifact"v012_data")
    artpath = artpaths[flux_version]
    xs = deserialize.(readdir(joinpath(artpath, "graphs"), join=true))
    
    local num_features
    if flux_version == "0.11"
        num_features = size(xs[1].features,1)
    elseif flux_version == "0.12"
        num_features = size(xs[1].encoded_features,1)
    end

    # build model
    local model
    if flux_version == "0.11"
        model = Xie_model(
            num_features,
            num_conv = num_conv,
            atom_conv_feature_length = hidden_layer_width,
            pooled_feature_length = (Int(hidden_layer_width / 2)),
            num_hidden_layers = num_hidden_layers,
        )
    elseif flux_version == "0.12"
        model = build_CGCNN(
            num_features,
            num_conv = num_conv,
            atom_conv_feature_length = hidden_layer_width,
            pooled_feature_length = (Int(hidden_layer_width / 2)),
            num_hidden_layers = num_hidden_layers,
        )
    end
    
    return xs, model
end

function atomicgraphnets_fw(flux_version; batchsize = 100, num_conv = 3, hidden_layer_width = 32, num_hidden_layers = 1)
    xs, model = agn_setup(flux_version, batchsize, num_conv, hidden_layer_width, num_hidden_layers)

    od_group["AtomicGraphNets_fw_Fluxv$(flux_version)_$(batchsize)pts_$(num_conv)conv_hlw$(hidden_layer_width)_hln$(num_hidden_layers)_lr$(lr)"] = b = @benchmarkable(model.(xs))
end

function atomicgraphnets_bw(flux_version, batchsize = 100, num_conv = 3, hidden_layer_width = 32, num_hidden_layers = 1, lr=0.001)
    xs, model = agn_setup(flux_version, batchsize, num_conv, hidden_layer_width, num_hidden_layers)

    od_group["AtomicGraphNets_bw_Fluxv$(flux_version)_$(batchsize)pts_$(num_conv)conv_hlw$(hidden_layer_width)_hln$(num_hidden_layers)_lr$(lr)"] = b = @benchmarkable(gradient.((x) -> model(x), xs))
end