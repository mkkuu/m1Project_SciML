using NPZ
using OrdinaryDiffEq
using DiffEqFlux
using Lux
using Random
using Optimisers
using Zygote
using DifferentialEquations
using Flux
using Statistics
using ProgressMeter
using ComponentArrays


# We open the reduced state previously prepared in Python
reducedState = npzread("data/processed/sstReducedStateCOPERNICUS20102019Prepared.npz")

# for key in keys(reducedState) 
#     println(key)
# end

# println(reducedState["tVal"])

# We set a shape of struct adapted to the input reduced state
struct inputVars
    PCsTrain
    PCsVal
    tTrain
    tVal
end

# We instance the retrieve .npz reduced state
iV = inputVars(reducedState["PCsTrain"], reducedState["PCsVal"], reducedState["tTrain"], reducedState["tVal"])

println(size(iV.PCsTrain)) # (2922,150)
println(size(iV.PCsVal)) # (730, 150)

# We remerge initial unsplitted PCs state
PCs = cat(iV.PCsTrain, iV.PCsVal, dims=1)
dT = iV.tTrain[2]-iV.tTrain[1] # We assume time step as already been normalized
spanT = size(PCs)[1]

println(spanT) # 3652
println(dT) # 1.0
println(size(PCs)) # (3652, 150) -> correct because we compute 150 mods and we have 10 years of data with one point for each day

# From now on, we desire to implement Neural ODE network

# Before we are splitting dataset between a train set and a validation set
ratioTrain = 0.7 # We set the ratio of the set we want to train (so we can easily modify it later)
zTrain, zTest = PCs[1:Int32.(round(ratioTrain*spanT)), :], PCs[Int32.(round(ratioTrain*spanT)):spanT, :] # Here we separete train and validation dataset

# println(size(zTrain))
# println(size(zTest))

zTrain = Float32.(zTrain)
zTest  = Float32.(zTest)

# Now we can start to set the NN and its parameters
z0 = Float32.(zTrain[1, :])
tSpan = (0, Int32.(round(ratioTrain*spanT)))
nMods = size(zTrain)[2]
# println(z0)
# println(tSpan)
println(nMods)

# We follow the exemplage of Lux.jl to construct a NN layer
rng = Xoshiro(0)

dZdT = Lux.Chain(
    Lux.Dense(nMods, 64, tanh),
    Lux.Dense(64, nMods)
)

ps, st = Lux.setup(rng, dZdT)
ps = ComponentArray(ps)


tspan = (0f0, Float32.(spanT - 1))

NODE = NeuralODE(
    dZdT,
    tspan,
    Tsit5(),
    saveat = 1f0
)

function predictNODE(z0, ps, st)
    sol, _ = NODE(z0, ps, st)
    Array(sol)
end

function lossNODE(ps)
    pred = predictNODE(z0, ps, st)
    return mean((pred .- zTrain').^2)
end

# To review (on how an optimiser really works)
opt = Optimisers.Adam(1e-3)
optState = Optimisers.setup(opt, ps)

function train!(ps, st, optState; nEpochs=200)
    @showprogress for epoch in 1:nEpochs
        loss, back = Zygote.pullback(lossNODE, ps)
        grads = back(1f0)[1]
        optState, ps = Optimisers.update(optState, ps, grads)

        epoch % 10 == 0 && println("Epoch $epoch | Loss = $loss")
    end
    return ps, optState
end

ps, optState = train!(ps, st, optState)










