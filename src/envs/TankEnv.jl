export TankEnv
export TankDynamics
using ReinforcementLearning
using Flux
using StableRNGs
using IntervalSets
using ControlSystems
using Distributions, Random

include("TankDynamics.jl")

"""
We distinguish between the tank environment and the tank dynamics:
- environment: parameters and methods that pertain to the "learning" aspect
- dynamics: the ground truth dynamic equations and parameters comprising the system
"""

struct TankEnvParams{T}
    max_steps::Int
    io_state::Bool
    L::Int
    N::Int
    L_q::Int
end

Base.show(io::IO, env_params::TankEnvParams) = print(
    io,
    join(["$p=$(getfield(env_params, p))" for p in fieldnames(TankEnvParams)], ","),
)

function TankEnvParams(;
    T = Float64,
    max_steps = 400,
    io_state = true,
    L = 8,
    N = 400,
    L_q = 8,
)
    TankEnvParams{T}(
        max_steps,
        io_state,
        L,
        N,
        L_q
    )
end

mutable struct TankEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    sys::TankDynamics{T}
    env_params::TankEnvParams{T}
    action_space::A
    observation_space::Space{Vector{ClosedInterval{T}}}
    state::Vector{T}
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
end

"""
    TankEnv(;kwargs...)

# Keyword arguments
- `T = Float64`
- `max_steps = 200`
"""
function TankEnv(;
    T = Float64,
    max_steps = 400,
    rng = Random.GLOBAL_RNG,
    io_state = true,
    L = 8,
    N = 400,
    L_q = 8,
    sp = 0.0,
)
    sys = TankDynamics(T=T, level_sp=10.0)
    reset(sys)
    

    env_params = TankEnvParams(T=T, max_steps=max_steps, io_state=io_state, L=L, N=N, L_q=L_q)
    action_space = -100.0..100.0

    if io_state == false
        low = repeat([typemin(T)], size(sys.A,1))
        high = repeat([typemax(T)], size(sys.A,1))
    else
        low = repeat([typemin(T)], 2*env_params.L_q+2)
        high = repeat([typemax(T)], 2*env_params.L_q+2)
    end
    state_space = Space(
        ClosedInterval{T}.(low, high),
    )

    probe_u = sign.(excite(L+1, num_data=N, m=1))
    pe_u, pe_y = tank_probe(sys, probe_u)
    sys.level_sp=sp
    reset(sys)

    H_u = Hankel(vec(pe_u[1:end-1]), L)
    H_y = Hankel(vec(pe_y[1:end-1]), L)
    H_u_full = Hankel(vec(pe_u[2:end]), L)
    H_y_full = Hankel(vec(pe_y[2:end]), L)
    H_inv = pinv([H_u; H_y])
    U = zeros(L^2 - 1)
    Y = zeros(L^2 - 1)

    env = TankEnv(
        sys,
        env_params,
        action_space,
        state_space,
        io_state ? zeros(T, 2*env_params.L_q+4) : zeros(T, size(sys.A,1)),
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
    )
    reset!(env)
    env
end

TankEnv{T}(; kwargs...) where {T} = TankEnv(T = T, kwargs...)

Random.seed!(env::TankEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::TankEnv) = env.action_space
RLBase.state_space(env::TankEnv) = env.observation_space
RLBase.reward(env::TankEnv{A,T}) where {A,T} = -(abs(env.state[end]))./10.0 - 0.01(env.action)^2 # - 0.1*(env.state[end-3]^2)
RLBase.is_terminated(env::TankEnv) = env.done
RLBase.state(env::TankEnv) = env.state

function RLBase.reset!(env::TankEnv{A,T}) where {A,T}

    env.sys = TankDynamics(T=T, level_sp=env.sp)
    env.sys.level_sp = 10.0

    if env.env_params.io_state
        env.state = zeros(T, 2*env.env_params.L_q+4)
    else
        env.state[:] = T(0.1) * rand(env.rng, T, size(env.sys.A,1)) .- T(0.05)
        env.x[:] = env.state[:]
    end
    env.state[env.env_params.L_q] = env.sp # needed to initialize the Qinput values: we assume system starts at steady state with output = Hankel_output  Qinput starts at such that sp - output + predicted_output = sp
    env.t = 0
    env.action = 0.1rand(env.rng, env.action_space)
    env.U = zeros(env.env_params.L^2 - 1)
    env.Y = zeros(env.env_params.L^2 - 1)
    env.done = false
    nothing
end

function (env::TankEnv)(a::AbstractFloat)
    @assert a in env.action_space
    env.action = a
    _step!(env, env.action)
end

function (env::TankEnv)(a::Vector{Float64})
    a = a[1]
    @assert a in env.action_space
    env.action = a
    _step!(env, env.action)
end

function _step!(env::TankEnv, a)
    

    u = env.state[env.env_params.L_q+4] + a + PID(env.sys, mode="level", return_delta=true)
    # u = a + PID(env.sys, mode="level", return_delta=false)

    append!(env.U, u) # Important to have this at the top since the assumed structure of run.jl starts with computing action, then state, ...

    y = step(env.sys, u)

    track_error = env.sys.level_sp - y[1] # not env.sp

    if env.env_params.io_state

        y_complete = dot(env.H_y_full[end:end,:], env.H_inv*vcat(env.U[end-env.env_params.L+1:end], env.Y[end-env.env_params.L+1:end]))

        input = track_error .+ y_complete[end]
        
        input_prev = env.state[env.env_params.L_q+3]
        env.state[env.env_params.L_q+1] = env.state[env.env_params.L_q+2]
        env.state[env.env_params.L_q+2] = input - input_prev
        env.state[env.env_params.L_q+3] = input
        env.state[env.env_params.L_q+4] = u
        error_vec = env.state[env.env_params.L_q+5:end]
        append!(error_vec, track_error)
        deleteat!(error_vec, 1)
        env.state[env.env_params.L_q+5:end] = error_vec

    else
        env.state = vec(env.x)
    end

    append!(env.Y,y)

    if env.t == 400
        env.sys.level_sp = 5.0
    end

    env.t += 1
    env.done = env.t >= env.env_params.max_steps
    nothing

end

function tank_sim(env::TankEnv{T}, input::Vector) where {T}
    inputs = Float64[]
    outputs = Float64[]
    rewards = Float64[]
    delta_u = 0.0
    for i in 1:length(input)-1
        _step!(env, delta_u)
        delta_u = input[i+1] - input[i]
        push!(inputs, env.U[end])
        push!(outputs, env.Y[end])
        push!(rewards, reward(env)[1])
    end
    return inputs, outputs, rewards
end
