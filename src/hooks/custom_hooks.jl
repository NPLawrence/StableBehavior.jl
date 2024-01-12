export EpisodeIO
export YKHook
export YK_state
export Stable_param

using ReinforcementLearning
using Plots
using Dates
using DataFrames
using CSV
using UnicodePlots
using UnicodePlots:lineplot, lineplot!

include("../policies/YK_param.jl")
include("../policies/stablePID.jl")

function get_stable_mat(YK::YK_param_stable)
    # stableMat(YK.M, YK.P)
    stableMat(YK.S, YK.U, YK.P, YK.V)
    # stableMat(YK.M)
end

Base.@kwdef mutable struct EpisodeIO <: AbstractHook
    input = Vector{Vector{Float64}}()
    output = Vector{Vector{Float64}}()
end

function (hook::EpisodeIO)(::PreEpisodeStage, agent, env)
    push!(hook.input, [])
    push!(hook.output, [])
    # push!(hook.YK_eigen, [])
end


function (hook::EpisodeIO)(::PostActStage, agent, env)
    push!(hook.input[end], env.env.U[end])
    push!(hook.output[end], env.env.Y[end])
end

function (hook::EpisodeIO)(::PostExperimentStage, agent, env)

    gr()

    p = lineplot(hook.output[end], title="Final rollout", xlabel="Time step", ylabel="Output")
    lineplot!(p, fill(env.env.sp, length(hook.output[end])))
    plt = plot(layout=(2,1))
    ylims!(plt[1], (minimum(vcat(hook.input...)), maximum(vcat(hook.input...))))
    ylims!(plt[2], (minimum(vcat(hook.output...)), maximum(vcat(hook.output...))))
    anim = @animate for (u,y) in zip(hook.input[1:2:end], hook.output[1:2:end])
        plot!(plt[1], u, label="", ylabel="Control input")
        plot!(plt[2], y, label="", ylabel="Process output", xlabel="Time")
    end
    gif(anim, fps=5)
    println(p)
end



Base.@kwdef mutable struct YKHook <: AbstractHook
    YK_eigen = Vector{Vector{ComplexF64}}()
end

function (hook::YKHook)(::PostEpisodeStage, agent, env)
    if Base.typename(typeof(agent.policy)).wrapper == TD3Policy
        YK = agent.policy.behavior_actor.model
    else
        YK = agent.policy.policy.model.μ
    end
    push!(hook.YK_eigen, eigen(get_stable_mat(YK)).values)
end

Base.@kwdef mutable struct YK_state <: AbstractHook
    x = Vector{}()
end

function (hook::YK_state)(::PreActStage, agent, env, action)
    n = size(agent.policy.behavior_actor.model.B,1)
    env.env.state[1:n] = vec(agent.policy.behavior_actor.model.x)
end

Base.@kwdef mutable struct StableParam <: AbstractHook
    sys::sys_data
    project::Bool = true
    PID_param = Vector{Vector{Float64}}()
    PID_param_project = Vector{Vector{Float64}}()
end

function (hook::StableParam)(::PostEpisodeStage, agent, env)
    K = vec(agent.policy.behavior_actor.model.K)
    push!(hook.PID_param, copy(K))
    if hook.project
        project_PID(hook::StableParam, agent, env, K, tol=0.0001)
    end
end

function (hook::StableParam)(::PreExperimentStage, agent, env)
    K = vec(agent.policy.behavior_actor.model.K)
    push!(hook.PID_param, copy(K))
    if hook.project
        project_PID(hook::StableParam, agent, env, K, tol=0.0001)
    end
end

function project_PID(hook::StableParam, agent, env, K; tol=0.05)
    ν = hook.sys.ν_design
    p = randn(3ν^2+6ν+6)
    # p = hook.sys.pinit .+ 0.1randn(3ν^2+6ν+6) # this might be faster but tends to be more brittle
    p[end-1:end] = K
    fstar, p_project, ret = project_sol(hook.sys, p, tol = tol)
    sleep(2)
    agent.policy.behavior_actor.model.K[end-1] = p_project[end-1]
    agent.policy.behavior_actor.model.K[end] = p_project[end]
    # hook.sys.pinit = p_project
    push!(hook.PID_param_project, p_project[end-1:end])
    println("\n", "fstar ", fstar, " ret ", ret, " kp,ki ", p_project[end-1:end])
end