using GeometricFlux
using GeometricFlux.Datasets

struct GCN end


function geometricflux_add_gcn(geoflux_group, ::Type{GCN}, hidden_dim, bch_sz)
    dataset = :cora
    input_dim, target_dim = 1433, 7

    train_X, _ = map(x -> Matrix(x), alldata(Planetoid(), dataset, padding=true))
    g = graphdata(Planetoid(), dataset)

    fg = FeaturedGraph(g)
    train_data = repeat(train_X, outer=(1,1,bch_sz))

    model = Chain(
        WithGraph(fg, GCNConv(input_dim=>hidden_dim, relu)),
        Dropout(0.5),
        WithGraph(fg, GCNConv(hidden_dim=>target_dim)),
    )

    geoflux_group["GeometricFlux_GCN_Forward_hidden_$(hidden_dim)_bch_sz_$(bch_sz)"] = @benchmarkable(
        fw(model, data),
        setup = (
            data = $train_data |> gpu;
            model = $model |> gpu
        ),
        teardown = (GC.gc(); CUDA.reclaim())
    )

    geoflux_group["GeometricFlux_GCN_Backward_hidden_$(hidden_dim)_bch_sz_$(bch_sz)"] = @benchmarkable(
        bw(model, data),
        setup = (
            data = $train_data |> gpu;
            model = $model |> gpu
        ),
        teardown = (GC.gc(); CUDA.reclaim())
    )
end

