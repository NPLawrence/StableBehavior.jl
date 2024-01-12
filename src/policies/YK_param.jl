"""
Code for a linear Youla-Kucera parameter (vs stable_policy.jl); not used in these examples, but maybe nice to have
"""

export YK_param

using Flux
using Flux: @functor, glorot_uniform, kaiming_normal
using ControlSystems
using Flux: params

include("stableMatrices.jl")

struct YK_param{F <: Function, S <: AbstractArray}
    A::S
    B::S
    C::S
    D::S
    σ::F
end

struct YK_param_stable{F <: Function, S <: AbstractArray}
    S::S
    U::S
    P::Vector
    V::S
    B::S
    C::S
    D::S
    σ::F
end


YK_param_stable(S, U, P, V, B, C, D) = YK_param_stable(S, U, P, V, B, C, D, identity)

@functor YK_param
@functor YK_param_stable

function YK_param(n::Integer, m::Integer, p::Integer, σ = identity; rng=MersenneTwister(123),
    initYK = glorot_uniform, stable = true)

    if stable
        return YK_param_stable(initYK(rng, n, n), initYK(rng, n, n), initYK(rng, n), initYK(rng, n, n), initYK(rng, n, m), initYK(rng, p, n), initYK(rng, p, m), σ)
    else
        return YK_param(initYK(rng, n, n), initYK(rng, n, m), initYK(rng, p, n), initYK(rng, p, m), σ)
    end

end

function act(A,B,C,D,x)

    n, m = size(A,2), size(B,2)

    PE_in, PE_out = SS_signal(A, B, C, D)
    H_in = Hankel(PE_in, n+1)
    H_out = Hankel(PE_out[1:end-1], n)
    H_out_full = Hankel(PE_out[2:end], n)

    Qinput = x[1:n+1,:]
    Qoutput = x[n+2:2(n+1)-1,:]
    u = *(H_out_full[end:end,:], pinv([H_in;H_out])*[Qinput; Qoutput])

    return u

end

function (model::YK_param)(x::AbstractArray)
    A, B, C, D, σ = model.A, model.B, model.C, model.D, model.σ
    
    u = act(A,B,C,D,x)
    
    return u
end

function (model::YK_param_stable)(x::AbstractArray)
    S, U, P, V, B, C, D, σ =  model.S, model.U, model.P, model.V, model.B, model.C, model.D, model.σ

    A = stableMat(S,U,P,V)
    u = act(A,B,C,D,x)

    return u
end
