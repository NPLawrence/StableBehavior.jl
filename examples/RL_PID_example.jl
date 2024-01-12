using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using ControlSystems
using Distributions, Random
using Flux: glorot_uniform, kaiming_normal
using GenericLinearAlgebra: svd
using Flux: params # BREAKING Flux >/ 0.13.0 doesn't seem to use `params()` anymore, causing issues with ReinforcementLearning.jl
using BSON
using BSON: @save, @load
using DelimitedFiles, CSV
# using Wandb, Dates, Logging
include("../src/envs/LTIEnvPID.jl")
include("../src/policies/PIDactor.jl")
include("../src/hooks/custom_hooks.jl")

"""
This training session is pretty expensive, but if you just want a fast run without constraining the PI parameters 
(maybe for rapid testing or if you have a good initialization), set the value below to `false`
"""
constrained_parameters = true

num_runs = 1
time = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
expdir = mkdir("./examples/experiments/testing-batch-$(time)")
mkdir(expdir*"/figures")
notes = ""
writedlm(expdir*"/note.txt", notes)

function sys_data(env::LTIEnvPID; ν_design::Int64=5)
    sys = env.env_params.sys
    u = vec(env.pe_u)
    y = vec(env.pe_y)
    ν = env.env_params.L
    L = ν + 1
    h = env.env_params.sys.Ts
    pinit = randn(3ν_design^2+6ν_design+6)
    H_y = Hankel(vec(y), L)
    H_u_init = Hankel(vec(u)[1:length(u) - L + ν], ν)
    H_y_init = Hankel(vec(y)[1:length(y) - L + ν], ν)
    H_pinv = pinv([H_u_init; H_y_init])
    sys_data(sys, u, y, ν, h, pinit, H_y, H_pinv, ν_design)
end

## Train a policy
function LTI_Experiment(;
    seed = 123,
    save_extras = true,
)
    rng = MersenneTwister(seed)
    n_q = 6

    inner_env = LTIEnvPID(L_q = n_q+1, max_steps = 100, L=11, N=400, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> 1.0x,
    )

    ns = length(state(inner_env))
    init = glorot_uniform(rng)

    create_actor() = PIDactor(identity, rng=MersenneTwister()) |> gpu

    create_critic_model() = Chain(
        Dense(ns + 1, 64, relu; init = init),
        Dense(64, 64, relu; init = init),
        Dense(64, 1; init = init),
    ) |> gpu

    create_critic() = TD3Critic(create_critic_model(), create_critic_model())

    agent = Agent(
        policy = TD3Policy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            # γ = 0.99f0,
            # ρ = 0.99f0,
            # batch_size = 64,
            start_steps = 50,
            start_policy = RandomPolicy(-1.0..1.0; rng = rng),
            update_after = 50,
            update_freq = 1,
            # policy_freq = 4,
            target_act_limit = high,
            # target_act_noise = 0.1,
            act_limit = high,
            # act_noise = 0.1,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10_000,
            state = Vector{Float64} => (ns,),
            action = Float64 => (),
        ),
    )

    stop_condition = StopAfterEpisode(20, is_show_progress=!haskey(ENV, "CI"))
    # stop_condition = StopAfterStep(20_000, is_show_progress=true)
    total_reward_per_episode = TotalRewardPerEpisode()
    episode_io = EpisodeIO()
    project_params = StableParam(sys=sys_data(inner_env), project=constrained_parameters)
    time = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")

    make_dir(path,n) = !isdir(path*"-$(n)/") ? mkdir(path*"-$(n)/") : make_dir(path, n+1)
    
    path = expdir*"/run-$(time)"
    rundir = make_dir(path,0)

    if save_extras
        hook = ComposedHook(
            total_reward_per_episode, 
            episode_io,
            project_params,
            DoOnExit() do agent, env
                CSV.write(rundir*"rewards.csv", DataFrame(reshape(total_reward_per_episode.rewards,1,:), :auto), header=[])
                CSV.write(rundir*"input_progress.csv", DataFrame(hcat(episode_io.input...), :auto),
                    header = [])
                CSV.write(rundir*"output_progress.csv", DataFrame(hcat(episode_io.output...), :auto),
                    header = [])
                CSV.write(rundir*"PID_param.csv", DataFrame(hcat(project_params.PID_param...), :auto),
                    header = [])
                CSV.write(rundir*"PID_param_project.csv", DataFrame(hcat(project_params.PID_param_project...), :auto),
                    header = [])
                Policy = agent.policy
                BSON.@save rundir*"policy.bson" Policy
                BSON.@save rundir*"env.bson" inner_env
            end
            )
    else
        hook = ComposedHook(
            total_reward_per_episode, 
            project_params,
            DoOnExit() do agent, env
                CSV.write(rundir*"rewards.csv", DataFrame(reshape(total_reward_per_episode.rewards,1,:), :auto), header=[])
                Policy = agent.policy
                BSON.@save rundir*"policy.bson" Policy
                BSON.@save rundir*"env.bson" inner_env
            end
            )
    end

    Experiment(agent, env, stop_condition, hook, "# LTI with TD3")
end

using Plots

ex(seed) = LTI_Experiment(seed=seed, save_extras=true)

function get_batch(num_runs = 1)
    seeds = rand(1:max(1000, num_runs), num_runs)
    [ex(seeds[i]) for i in 1:num_runs]
end

function batch_experiment(Expts::Vector{Experiment})
    i = 0
    for ex in Expts
        i += 1
        println("Running experiment #$(i)")
        run(ex)
    end
    
end

Expts = get_batch(num_runs)
batch_experiment(Expts)
