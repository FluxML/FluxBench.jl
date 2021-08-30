using Pkg; Pkg.activate(@__DIR__)
using Pkg.Artifacts
using ChemistryFeaturization
using AtomicGraphNets
using Serialization
using CSV, DataFrames
using Flux

# general options, hyperparams, etc.
train_frac = 0.8
num_conv = 3 # how many convolutional layers?
crys_fea_len = 32 # length of crystal feature vector after pooling
num_hidden_layers = 1 # how many fully-connected layers after convolution and pooling?
opt = Flux.Optimise.ADAM(0.001) # optimizer

# load up the data (it's pre-shuffled)
artpath = artifact"v012_data"
info = CSV.read(joinpath(artpath, "info.csv"), DataFrame)
x = FeaturizedAtoms{AtomGraph,GraphNodeFeaturization}[]
for r in eachrow(info)
    fname = r[:task_id]*".jls"
    fa = deserialize(joinpath(artpath, "graphs", fname))
    push!(x, fa)
end
y = info.formation_energy_per_atom
num_features = size(x[1].encoded_features,1)

# split into train/test
num_train = Int(round(train_frac*length(x)))
x_train = x[1:num_train]
y_train = y[1:num_train]
train_data = zip(x_train, y_train)
x_test = x[num_train+1:end]
y_test = y[num_train+1:end]

# build model
model = build_CGCNN(
    num_features,
    num_conv = num_conv,
    atom_conv_feature_length = crys_fea_len,
    pooled_feature_length = (Int(crys_fea_len / 2)),
    num_hidden_layers = 1,
)
loss(x, y) = Flux.Losses.mse(model(x), y)
