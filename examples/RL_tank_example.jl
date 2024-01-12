using ReinforcementLearning
using StableRNGs
using Flux
using Flux.Losses
using ControlSystems
using Distributions, Random
using Flux: glorot_uniform, kaiming_normal
using Flux: params # BREAKING Flux >/ 0.13.0 doesn't seem to use `params()` anymore, causing issues with ReinforcementLearning.jl
using BSON
using BSON: @save, @load
using DelimitedFiles, CSV
# using Wandb, Dates, Logging
include("../src/envs/TankEnv.jl")
include("../src/policies/stable_policy.jl")
include("../src/hooks/custom_hooks.jl")

num_runs = 1
time = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")
expdir = mkdir("./examples/experiments/testing-batch-$(time)")
mkdir(expdir*"/figures")
notes = ""
writedlm(expdir*"/note.txt", notes)
## Train a policy
function Tank_Experiment(;
    seed = 123,
    save_extras = true,
)
    rng = MersenneTwister(seed)
    n_q = 5

    inner_env = TankEnv(io_state = true, L_q = n_q, max_steps = 800, L=11, N=500, rng = rng)
    A = action_space(inner_env)
    low = A.left
    high = A.right
    env = ActionTransformedEnv(
        inner_env;
        action_mapping = x -> 1.0x,
    )

    ns = length(state(inner_env))
    init = glorot_uniform(rng)

    create_actor() = stable_policy(n_q, n_q+4, 16, 2) |> gpu
    # create_actor() = Chain(
    #     Dense(2n_q+4, 64, relu; init = init),
    #     Dense(64, 64, relu; init = init),
    #     Dense(64, 1; init = init),
    # ) |> gpu

    create_critic_model() = Chain(
        Dense(ns + 1, 64, softplus; init = init),
        Dense(64, 64, softplus; init = init),
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
            γ = 0.99f0,
            ρ = 0.99f0,
            batch_size = 64,
            start_steps = 100,
            start_policy = RandomPolicy(-1.0..1.0; rng = rng),
            update_after = 100,
            update_freq = 1,
            policy_freq = 4,
            target_act_limit = high,
            target_act_noise = 0.5,
            act_limit = high,
            act_noise = 0.5,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 10_000,
            state = Vector{Float64} => (ns,),
            action = Float64 => (),
        ),
    )

    stop_condition = StopAfterEpisode(100, is_show_progress=!haskey(ENV, "CI"))
    total_reward_per_episode = TotalRewardPerEpisode()
    episode_io = EpisodeIO()
    Q_state = YK_state()
    time = Dates.format(now(),"yyyy_mm_dd_HH_MM_SS")

    make_dir(path,n) = !isdir(path*"-$(n)/") ? mkdir(path*"-$(n)/") : make_dir(path, n+1)
    
    path = expdir*"/run-$(time)"
    rundir = make_dir(path,0)

    if save_extras
        hook = ComposedHook(
            total_reward_per_episode, 
            episode_io,
            Q_state,
            DoOnExit() do agent, env
                CSV.write(rundir*"rewards.csv", DataFrame(reshape(total_reward_per_episode.rewards,1,:), :auto), header=[])
                CSV.write(rundir*"input_progress.csv", DataFrame(hcat(episode_io.input...), :auto),
                    header = [])
                CSV.write(rundir*"output_progress.csv", DataFrame(hcat(episode_io.output...), :auto),
                    header = [])
                Policy = agent.policy
                BSON.@save rundir*"policy.bson" Policy
                BSON.@save rundir*"env.bson" inner_env
            end
            )
    else
        hook = ComposedHook(
            total_reward_per_episode, 
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

ex(seed) = Tank_Experiment(seed=seed, save_extras=true)

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
