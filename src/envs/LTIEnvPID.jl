export LTIEnvPID

using ReinforcementLearning
using Flux
using StableRNGs
using IntervalSets
using ControlSystems
using Distributions, Random

struct LTIEnvPIDParams{T}
    sys::StateSpace{ControlSystems.Discrete{T}, T}
    max_steps::Int
    L::Int
    N::Int
    L_q::Int
end

Base.show(io::IO, env_params::LTIEnvPIDParams) = print(
    io,
    join(["$p=$(getfield(env_params, p))" for p in fieldnames(LTIEnvPIDParams)], ","),
)

function LTIEnvPIDParams(;
    T = Float64,
    sys = ss(c2d(tf([-1, 1],[1, 3, 3, 1]), 0.5)),
    max_steps = 200,
    L = 4,
    N = 100,
    L_q = 4,
)
    LTIEnvPIDParams{T}(
        sys,
        max_steps,
        L,
        N,
        L_q
    )
end

mutable struct LTIEnvPID{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    env_params::LTIEnvPIDParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
    x::Vector{T}
    action::ACT
    pe_u::VecOrMat{T}
    pe_y::VecOrMat{T}
    H_u::VecOrMat{T}
    H_y::VecOrMat{T}
    H_u_full::VecOrMat{T}
    H_y_full::VecOrMat{T}
    H_inv::VecOrMat{T}
    U::Vector{T}
    Y::Vector{T}
    done::Bool
    t::Int
    rng::R
    sp::Float64
    var::Float64
end

"""
    LTIEnvPID(;kwargs...)

# Keyword arguments
- `T = Float64`
- `max_steps = 200`
"""
function LTIEnvPID(;
    T = Float64,
    sys = ss(c2d(tf([-1, 1],[1, 3, 3, 1]), 0.5)),
    max_steps = 200,
    rng = Random.GLOBAL_RNG,
    L = 4,
    N = 100,
    L_q = 4,
    sp = 1.0,
    var = 0.01,
)
    env_params = LTIEnvPIDParams(T=T, sys=sys, max_steps=max_steps, L=L, N=N, L_q=L_q)
    action_space = -10.0..10.0

    low = repeat([typemin(T)], 3)
    high = repeat([typemax(T)], 3)
    state_space = Space(
        ClosedInterval{T}.(low, high),
    )

    pe_u = excite(L+1, num_data=N, m=size(sys.B,2))
    pe_y = lsim(sys, pe_u').y .+ var*randn(size(pe_u'))

    H_u = Hankel(vec(pe_u[1:end-1]), L)
    H_y = Hankel(vec(pe_y[1:end-1]), L)
    H_u_full = Hankel(vec(pe_u[2:end]), L)
    H_y_full = Hankel(vec(pe_y[2:end]), L)
    H_inv = pinv([H_u; H_y])
    U = zeros(L^2 - 1)
    Y = zeros(L^2 - 1)

    env = LTIEnvPID(
        env_params,
        action_space,
        state_space,
        zeros(T, 3),
        zeros(T, size(env_params.sys.A,1)),
        0.0,
        pe_u,
        pe_y,
        H_u,
        H_y,
        H_u_full,
        H_y_full,
        H_inv,
        U,
        Y,
        false,
        0,
        rng,
        sp,
        var
    )
    reset!(env)
    env
end

LTIEnvPID{T}(; kwargs...) where {T} = LTIEnvPID(T = T, kwargs...)

Random.seed!(env::LTIEnvPID, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::LTIEnvPID) = env.action_space
RLBase.state_space(env::LTIEnvPID) = env.observation_space
RLBase.reward(env::LTIEnvPID{A,T}) where {A,T} = (abs(env.state[2])./env.env_params.sys.Ts) < 0.50 # -(abs(env.state[2])^2)
RLBase.is_terminated(env::LTIEnvPID) = env.done
RLBase.state(env::LTIEnvPID) = env.state

function RLBase.reset!(env::LTIEnvPID{A,T}) where {A,T}
    env.state[:] = zeros(T, 3, 1)
    env.x[:] = zeros(T, size(env.env_params.sys.A,1))

    env.t = 0
    env.action = 0.0
    env.sp = rand([2.0, -2.0])
    env.U = zeros(env.env_params.L^2 - 1)
    env.Y = zeros(env.env_params.L^2 - 1)
    env.done = false
    nothing
end

function (env::LTIEnvPID)(a::AbstractFloat)
    @assert a in env.action_space
    env.action = a
    _step!(env, env.action)
end

function (env::LTIEnvPID)(a::Vector{Float64})
    a = a[1]
    @assert a in env.action_space
    env.action = a
    _step!(env, env.action)
end

function _step!(env::LTIEnvPID, a)
    
    env.t += 1

    env.x = env.env_params.sys.A*env.x + vec(env.env_params.sys.B*a)
    y = dot(env.env_params.sys.C, env.x) + env.var*randn() # assume no D matrix

    track_error = env.sp - y[1]

    # state has the form (Δe, Δt e, u_prev)
    Ts = env.env_params.sys.Ts
    e_prev = env.state[2] / Ts
    env.state[1] = track_error - e_prev
    env.state[2] = Ts*track_error
    env.state[3] = a

    append!(env.U, a)
    append!(env.Y,y)

    env.done = env.t >= env.env_params.max_steps
    nothing

end

