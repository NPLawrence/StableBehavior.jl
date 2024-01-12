export PIDactor

using Flux
using Flux: @functor, glorot_uniform, kaiming_normal
using ControlSystems
using Flux: params

struct PIDactor{F <: Function, S <: AbstractArray}
    K::S
    σ::F
end

PIDactor(K) = PIDactor(K, identity)

@functor PIDactor

function PIDactor(σ = identity; rng=MersenneTwister(123),
    init = glorot_uniform)

    return PIDactor(0.1init(rng, 2, 1), σ)
end

function (model::PIDactor)(x::AbstractArray)
    *(vcat(model.K, 1.0)', x)
end
