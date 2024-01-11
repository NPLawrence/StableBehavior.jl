export stable_policy

using Flux
using Flux: @functor, glorot_uniform, kaiming_normal
using ControlSystems
using Flux: params
using Plots
using ColorSchemes

include("ICNN.JL")

mutable struct stable_policy{F<:Chain, S <: AbstractArray}
    f::F
    V::ICNN
    B::S 
    C::S 
    D::S 
    x::Matrix
end

@functor stable_policy

function ReHU(x)
    d = 1.0
    max.(x.-d/2, clamp.(sign.(x).*x.^2 ./ (2*d),0.0,d/2.0))
end

function stable_policy(n::Integer, k::Integer, m::Integer, l::Integer; σ=tanh, init=Flux.kaiming_normal)
    layers = [ Dense(m, m, σ; init = init) for i in 1:l]
    f = Chain(vcat(Dense(n+k, m, σ; init = init), layers, Dense(m, n; init = init))...)
    V = ICNN(n, m, l, ReHU)
    B = 0.1init(n,1)
    C = 0.1init(1,n)
    D = 0.1init(1,1)
    stable_policy(f, V, B, C, D, zeros(n,1))
end

function lyapunov(model::stable_policy, x::AbstractArray)
    ReHU.(model.V(x) .- model.V(zeros(size(x)))) .+ 0.1*norm(x)^2 
end

function (model::stable_policy)(x::AbstractArray)
    β = 0.99
    n = size(model.x,1)
    x_Q = x[1:n,:]
    Vx = lyapunov(model, x_Q)
    Vf = lyapunov(model, model.f(x))

    # This is the simple-but-not-quite-right implementation as it potentially includes division by zero
    # according to the actual formulation, if Vf = 0, then we don't need scaling (since Vf ≤ β Vx), so the output should just be f(x)
    # better would be to implement piecewise
    x_f = model.f(x).*(β*Vx .- relu.(β*Vx .- Vf)) ./ Vf

    input_prev_Q = x[n+1:n+1,:]
    input_Q = x[n+2:n+2,:]

    x_Q = x_f + model.B*input_prev_Q
    u = model.C*x_Q + model.D*input_Q

    if size(x_Q,2)==1
        model.x = x_Q
    end
    return u
end