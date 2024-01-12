using Plots
import .Plots.gui
import .Plots.plot
import .Plots.plot!
import .Plots.annotate!
using ColorSchemes

using Flux
using ControlSystems
using StatsPlots
using Distributions
using LaTeXStrings
using PGFPlotsX
using RollingFunctions
using ImageFiltering
# using IterTools

include("data_tools.jl")
include("../policies/YK_param.jl")
include("LTIEnv.jl")

trueBlues = ColorScheme([colorant"white"; ColorSchemes.Blues.colors])
# redblue = ColorScheme([reverse(ColorSchemes.Reds.colors); colorant"white"; colorant"white"; ColorSchemes.Blues.colors])
trueredbluewhite = ColorScheme([
    colorant"#a90c38",
    colorant"#ec534b",
    colorant"#feaa9a",
    colorant"#ffffff",
    colorant"#ffffff",
    colorant"#9ac4e1",
    colorant"#5c8db8",
    colorant"#2e5a87",
])



function getSS(policy::YK_param)
    A, B, C, D = Flux.params(policy)
end

function getSS(policy::YK_param_stable)
    # S, U, P, V, B, C, D = Flux.params(policy)
    M, B, C, D = Flux.params(policy)

    # S, U, P, V,  = posdefMat(S), qr(U).Q, Diagonal(sigmoid.(P)), qr(V).Q
    # stableMat(S,U,P,V), B, C, D
    stableMat(M), B, C, D

end


function plot_env(env::LTIEnv, policy; kwargs...)
    gr()
    reset!(env)
    max_steps = env.env_params.max_steps
    Ts = env.env_params.sys.Ts

    p0 = plot(
        xlims=(0, env.env_params.max_steps),
        legend=true,
    )

    ## Plot
    outputs = Any[]
    for i in 1:max_steps
        a = policy(state(env))[]

        env(a)

        y = env.Y[end]
        append!(outputs, y)        
    end

    A, B, C, D = getSS(policy)
    
    Q = ss(A,B,C,D,Ts)

    C_Q = Q*feedback(1, -env.env_params.sys*Q)
    CL_Q = feedback(C_Q*env.env_params.sys)

    tf_y, tf_t, tf_x = lsim(CL_Q, (x,i)-> 1, 0:Ts:Ts*(max_steps-1))

    println("Q plot ", Q)
    println("eig A ", eigvals(A))
    println("Terminal reward: ", reward(env))

    plot!(p0, 0:max_steps-1, outputs, label="Hankel", linetype=:step)
    plot!(p0, 0:max_steps-1, tf_y', label="TF", linestyle=:dash, linetype=:step)
    plot!([1], seriestype = :hline, label="setpoint", color="orange", title="Closed-loop output")

    tf_inputs, tf_t_inputs, tf_x_inputs = lsim(C_Q/(1 + C_Q*env.env_params.sys), (x,i)-> 1, 0:Ts:Ts*(max_steps-1))

    p1 = plot(
        xlims=(0, env.env_params.max_steps),
        legend=false,
        xlabel="Time steps"
    )
    plot!(p1, 0:max_steps-1, tf_inputs', linetype=:step, title="Closed-loop input")

    p = plot(p0, p1, layout = (2,1))
    display(p)

end

function make_GIF(df, name="testing")
# makes GIF of training progress given a dataframe of rollouts

    min_df = min(describe(df).min...)
    max_df = max(describe(df).max...)
    cols = collect(eachcol(df))
    plt = plot(xlims=(0, nrow(df)), ylims=(min_df, max_df))

    anim = @animate for y in cols
        plot!(plt, y, label=false)
    end

    gif(anim, ("./figures/"*name*".gif"), fps=10)

end

function make_heatmap(df; d=35, grid_lines=false, save=false, save_path="./figures/")
# makes heatmap of training progress, or sequence of rollouts more generally
    pgfplotsx()

    # pyplot()
    # df = df[!,1:50]
    colors = palette(:default)
    A = collect(eachcol(df))

    min_df = min(describe(df).min...)
    max_df = max(describe(df).max...)
    
    D = Any[]

    levels = range(min_df, max_df, d)
    # levels = range(0, max_df, d)
    println(levels)

    for (i,l) in enumerate(levels[1:end-1])

        push!(D, count.(x -> l≤x<levels[i+1], A))

    end 

    # totals = log1p.(hcat(D...)) ./ maximum(log10.(1.0 .+ hcat(D...)))
    totals = log10.(1.0 .+ hcat(D...))
    # totals = zeros(size(totals))
    # totals = hcat(D...)
    println(maximum(hcat(D...)))
    # colorbar_style = PGFPlotsX.Options("ytick" => "{}" )
    # colorbar_style = PGFPlotsX.Options(
    #               "ytick" => "{"*prod(["$(tick)," for tick in [0.0, 0.1, 1.0]])[1:end-1]*"}",
    #               "yticklabels" => "{"*prod([L"10^0", L"10^1", L"10^2"].*",")[1:end-1]*"}")

    
    plt = heatmap(1:ncol(df), levels[1:end-1], totals', c=cgrad(trueBlues), colorbar_ticks=[log10(1+1), log10(10+1), log10(100+1)]) # note these are tick values are EXPONENTS
    direc = split(save_path,"/")[end-2]
    PGFPlotsX.push_preamble!(plt.o.the_plot, "% EXPERIMENT $(direc)")
    # plot!(aspect_ratio=1.0)
    # plt.get_colorbar_ticks()
    if grid_lines
        for i in 1:ncol(df)
            plot!([i+0.5], seriestype=:vline, lw=1.0, color=:white, label="")
        end
    end
    plot!([1], seriestype=:hline, linestyle=:dash, lw=4.0, label="", color=colors[2])
    # plot!(background_color=:white)
    xaxis!("Episode")
    yaxis!("Output")
    # plot!(plt, colorbar_ticks=[0.0, 1.0, 2.0, 10.0, 100.0])
    # Colorbar(plt, ticks=0:2)
    # if save
    #     savefig("./figures/"*name*".pdf")
    #     Plots.tex(plt, "./figures/"*name)
    # end
    savefig(save_path*".pdf")
    Plots.tex(plt, save_path)
    # gui()
    # plt
end

function make_heatmap(path::String; file::String="output_progress.csv", name="heatmap_output", d=100, grid_lines=true, save=true, types=Float64)
    # makes heatmap of training progress, or sequence of rollouts more generally
    pgfplotsx()
    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    stats = batch_stats(data)
    save_path = path*"/figures/"*name

    make_heatmap(DataFrame(reshape(stats["mean"], size(data[1])), :auto), save_path=save_path)

    # nothing
    # savefig(path*"/figures/"*name*".pdf")

end

function drawCircle(h, k, r; θ_max=2π)
    θ = LinRange(0, θ_max, 500)
    h .+ r*sin.(θ), k .+ r*cos.(θ)
end

function YK_eigen_scatter(df; name="testing_heatmap", display_plot=false)
    # gr()
    YK_eigen = vcat(eachcol(Matrix(df)')...)
    real_lim = -1.0:0.01:1.0
    im_lim = -1.0:0.01:1.0

    tol = 0.1

    D = Any[]

    # for z in Base.Iterators.product(real_lim, im_lim)
    #     # println(norm((real(YK_eigen[1]), imag(YK_eigen[1])).-z))
    #     # println(count.(x -> norm((real(x), imag(x)).-z)≤tol, YK_eigen))
    #     push!(D, count.(x -> norm((real(x), imag(x)).-z)≤tol, YK_eigen))
    # end
    # println(D)
    
    plt = plot()
    # heatmap(plt, real_lim, im_lim, (z1,z2) -> count(x -> norm((real(x), imag(x)).-(z1,z2))≤tol, YK_eigen),  c=cgrad(trueBlues), linewidth=0)

    ## For different colors
    # for i in 1:nrow(df)
    #     # println(size(row))
    #     plot!(plt, YK_eigen[ncol(df)*(i-1)+1:ncol(df)*(i)], seriestype=:scatter)
    # end
    plot!(plt, YK_eigen, seriestype=:scatter)


    plot!(plt, drawCircle(0, 0, 1), seriestype=:shape, lw=0.5, c=:blue, linecolor=:black, legend=false, fillalpha=0.0, aspect_ratio=1.0)
    # gui()
    if display_plot
        plot!(aspect_ratio=1.0, xlims=(-1.0,1.0), ylims=(-1.0,1.0))
        display(plt)
    else
        plt
    end
end


function YK_eigen_heatmap(P::Distribution; name="testing_heatmap", θ_max=2π, progress_bar=false, display_plot=true)
    gr()
    # YK_eigen = vcat(eachcol(df)...)

    real_lim = -1.0:0.01:1.0
    im_lim = -1.0:0.01:1.0

    # tol = 0.1

    plt = plot(aspect_ratio=1.0, xlims=(-1.0,1.0), ylims=(-1.0,1.0), showaxis=false, legend=false, xticks=[], yticks=[])
    # mixtures = Distributions[]
    # for row in eachrow(df)
    #     # println(vec(collect(row)))
    #     # println(real.(vec(collect(row))))
    #     # println(imag.(vec(collect(row))))
    #     mvs = [MvNormal([real(μ), imag(μ)], 0.01I) for μ in vec(collect(row))]
    #     # println(mvs)
    #     # println(mvs)
    #     # println(size(mvs))
    #     push!(mixtures, mvs)
    #     # heatmap!(real_lim, im_lim, (x,y) -> (pdf(mixmv,[x,y])),  c=cgrad(trueBlues), aspect_ratio=1.0)
    # end
    # for μ in vcat(eachcol(df)...)
    #     push!(mixtures, MvNormal([real(μ), imag(μ)]))
    # end
    # println(size(mixtures))
    # println(typeof(mixtures))
    # mixtures = [mix in mixtures]
    # if var_df === nothing
    #     contourf!(real_lim, im_lim, (x,y) -> pdf(MixtureModel([MixtureModel(map(u -> MvNormal([real(u[2]), imag(u[2])], (1/sqrt(u[1]))*0.1*I), enumerate(vcat(row...)))) for row in eachrow(df) ] ) ,[x,y]),  c=cgrad(trueBlues), aspect_ratio=1.0)
    # else
    contourf!(real_lim, im_lim, (x,y) -> pdf(P ,[x,y]),  c=cgrad(trueBlues), aspect_ratio=1.0, colorbar=false)

        # contourf!(real_lim, im_lim, (x,y) -> pdf(MixtureModel([MixtureModel(map(z -> MvNormal([real(z[1]), imag(z[1])], z[2]*I), zip(vcat(row...), vcat(std...)))) for (row, std) in zip(eachrow(df), eachrow(var_df)) ] ) ,[x,y]),  c=cgrad(trueBlues), aspect_ratio=1.0)
    # end
    # contourf!(plt, real_lim, im_lim, (z1,z2) -> count(x -> norm((real(x), imag(x)).-(z1,z2))≤tol, YK_eigen),  c=cgrad(trueBlues), linewidth=0)
    if progress_bar
        plot!(plt, drawCircle(0, 0, 1), lw=0.90, linecolor=:black, seriesalpha=0.5)
        plot!(plt, drawCircle(0, 0, 1, θ_max=θ_max), lw=1.0, linecolor=:black)
    else
        plot!(plt, drawCircle(0, 0, 1), lw=1.0, linecolor=:black)
    end
    # gui()
    if display_plot
        display(plt)
    end
    plt
end


# function YK_eigen_heatmap(df; var_df=nothing, name="testing_heatmap")



#     display(plt)
# end

function GIF_YK_eigen_heatmap(path::String; file::String="YK_eigen_progress.csv", types=ComplexF64, fps=6, local_iters=true, progress_bar=true, display_plot=true)
    gr()
    paths = batch_paths(path, file)
    savedir = isdir(path*"/figures/") ? path*"/figures/" : mkdir(path*"/figures/")
    data = batch_data(paths, types=types)
    n = size(data[1], 2)
    m = size(data[1], 1)
    F = Any[]
    anim = Animation()
    iters = local_iters ? n-m+1 : n

    for i in 1:iters
        if local_iters
            new_batch = [d[:, i:i+m-1] for d in data]
        else
            new_batch = [d[:, 1:i] for d in data]
        end
        stack = stack_batch(new_batch)
        var = knn_var(stack)
        mixture = get_mixture(stack, var=var)

        plt = YK_eigen_heatmap(mixture, θ_max = -(2π/iters)*i, progress_bar=progress_bar, display_plot=display_plot)
        if i==iters 
            savefig(savedir*"eigen_heatmap.png")
        end

        # plot!(plt, plt1)
        frame(anim, plt)
        push!(F, plt)

    end
    for f in reverse(F)
        frame(anim, f)
    end
    # anim2 = append(frame(anim), reverse(frame(anim)))
    # append(frame(anim), reverse(frame(anim)))
    
    gif(anim, savedir*"eigen_heatmap_local.gif", fps=fps)

end

function batch_rewards(path::String; file::String="rewards.csv", types=Float64, display_plot=false)
    pgfplotsx()
    direc = split(path,"/")[end]
    colors = palette(:default)

    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    stats = batch_stats(data)
    # upperbound=[-33.64]
    # lowerbound=[-102.59]
    # upperbound=[-33.64]
    # lowerbound=[-165.0]

    # plt_median = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    # PGFPlotsX.push_preamble!(plt_median.o.the_plot, "% EXPERIMENT $(direc)")
    # plot!(plt_median, stats["median"], ribbon=stats["iqr"], label="")
    # plot!(plt_median, upperbound, seriestype=:hline, linestyle=:dash, label="", color=colors[2])
    # plot!(plt_median, lowerbound, seriestype=:hline, linestyle=:dot, label="", color=colors[2])
    # savefig(path*"/figures/"*"rewards_median_iqr.pdf")
    # Plots.tex(plt_median,path*"/figures/"*"rewards_median_iqr.tex")
    # display(plt_median)

    # plt_quant_sym = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    # PGFPlotsX.push_preamble!(plt_quant_sym.o.the_plot, "% EXPERIMENT $(direc)")
    # plot!(plt_quant_sym, stats["median"], ribbon=stats["quant-range"], label="")
    # plot!(plt_quant_sym, upperbound, seriestype=:hline, linestyle=:dash, label="", color=colors[2])
    # plot!(plt_quant_sym, lowerbound, seriestype=:hline, linestyle=:dot, label="", color=colors[2])
    # savefig(path*"/figures/"*"rewards_median_quant_sym.pdf")
    # Plots.tex(plt_quant_sym,path*"/figures/"*"rewards_median_quant_sym.tex")
    # display(plt_quant)

    plt_quant = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    PGFPlotsX.push_preamble!(plt_quant.o.the_plot, "% EXPERIMENT $(direc)")
    plot!(plt_quant, rollmean(stats["median"], 2), ribbon=rollmean.((abs.(stats["median"] .- stats["quants"][2]), abs.(stats["median"] .- stats["quants"][1])), 2), label="")
    # plot!(plt_quant, stats["mean"], linestyle=:dash, label="")
    # plot!(plt_quant, upperbound, seriestype=:hline, linestyle=:dash, label="", color=colors[2])
    # plot!(plt_quant, lowerbound, seriestype=:hline, linestyle=:dot, label="", color=colors[2])

    savefig(path*"/figures/"*"rewards_median_quant.pdf")
    Plots.tex(plt_quant,path*"/figures/"*"rewards_median_quant.tex")
    # display(plt_quant)

    plt_mean = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    PGFPlotsX.push_preamble!(plt_mean.o.the_plot, "% EXPERIMENT $(direc)")
    plot!(plt_mean, stats["mean"], label="", ribbon=stats["std"])
    # plot!(plt_mean, upperbound, seriestype=:hline, linestyle=:dash, label="", color=colors[2])
    # plot!(plt_mean, lowerbound, seriestype=:hline, linestyle=:dot, label="", color=colors[2])
    # plot!(plt, stats["mean"], label="", ribbon=(stats["max], stats["min"]))
    savefig(path*"/figures/"*"rewards_mean_std.pdf")
    Plots.tex(plt_mean,path*"/figures/"*"rewards_mean_std.tex")
    # display(plt_mean)

    plt_mean_minmax = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward", xaxis=:log10)
    PGFPlotsX.push_preamble!(plt_mean_minmax.o.the_plot, "% EXPERIMENT $(direc)")
    plot!(plt_mean_minmax, stats["median"], label="", ribbon=(abs.(stats["median"].-stats["min"]), abs.(stats["max"].-stats["median"])), color=colors[1], fillalpha=0.20)
    plot!(plt_mean_minmax, stats["median"], ribbon=(abs.(stats["median"] .- stats["quants"][2]), abs.(stats["median"] .- stats["quants"][1])), label="", color=colors[1])
    # plot!(plt_mean_minmax, upperbound, seriestype=:hline, linestyle=:dash, label="", color=colors[2])
    # plot!(plt_mean_minmax, lowerbound, seriestype=:hline, linestyle=:dot, label="", color=colors[2])

    savefig(path*"/figures/"*"rewards_median_minmax.pdf")
    Plots.tex(plt_mean_minmax,path*"/figures/"*"rewards_median_minmax.tex")
    # display(plt_mean_minmax)

    # Plots.tex(plt, "./figures/"*name)
    if display_plot
        gui()
    end
end


function batch_rewards_PID(path1::String, path2::String; file::String="rewards.csv", types=Float64, display_plot=false)
    pgfplotsx()
    direc1 = split(path1,"/")[end]
    direc2 = split(path1,"/")[end]

    colors = palette(:default)

    paths1 = batch_paths(path1, file)
    data1 = batch_data(paths1, types=types)
    stats1 = batch_stats(data1)

    paths2 = batch_paths(path2, file)
    data2 = batch_data(paths2, types=types)
    stats2 = batch_stats(data2)

    meanparam = 2
    lw = 2
    plt_quant = plot(xlims=(1, length(data1[1])), xlabel="Episode", ylabel="Total reward")
    PGFPlotsX.push_preamble!(plt_quant.o.the_plot, "% EXPERIMENT $(direc1) and $(direc2)")
    plot!(plt_quant, rollmean(stats1["median"], meanparam), ribbon=rollmean.((abs.(stats1["median"] .- stats1["quants"][2]), abs.(stats1["median"] .- stats1["quants"][1])), meanparam), lw=lw, label="Stabilizing")
    plot!(plt_quant, rollmean(stats2["median"], meanparam), ribbon=rollmean.((abs.(stats2["median"] .- stats2["quants"][2]), abs.(stats2["median"] .- stats2["quants"][1])), meanparam), lw=lw, linestyle=:dash, label="Unconstrained")
    plot!(legend=:topleft)
    savefig(path1*"/figures/"*"rewards_median_quant.pdf")
    Plots.tex(plt_quant,path1*"/figures/"*"rewards_median_quant.tex")
    
    if display_plot
        gui()
    end
end


function batch_rewards_gr(path::String; file::String="rewards.csv", types=Float64, display_plot=false, label="")
    gr()
    direc = split(path,"/")[end]
    colors = palette(:default)

    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    stats = batch_stats(data)

    plt_quant = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    plot!(plt_quant, rollmean(stats["median"], 2), ribbon=rollmean.((abs.(stats["median"] .- stats["quants"][2]), abs.(stats["median"] .- stats["quants"][1])), 2), label=label, lw=2)
    # plot!(plt_quant, stats["mean"], linestyle=:dash, label="")

    savefig(path*"/figures/"*"rewards_median_quant.pdf")
    # display(plt_quant)

    plt_mean = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward")
    plot!(plt_mean, stats["mean"], label=label, ribbon=stats["std"])

    savefig(path*"/figures/"*"rewards_mean_std.pdf")
    # display(plt_mean)

    plt_mean_minmax = plot(xlims=(1, length(data[1])), xlabel="Episode", ylabel="Total reward", xaxis=:log10)
    plot!(plt_mean_minmax, stats["median"], label="", ribbon=(abs.(stats["median"].-stats["min"]), abs.(stats["max"].-stats["median"])), color=colors[1], fillalpha=0.20)
    plot!(plt_mean_minmax, stats["median"], ribbon=(abs.(stats["median"] .- stats["quants"][2]), abs.(stats["median"] .- stats["quants"][1])), label=label, color=colors[1])

    savefig(path*"/figures/"*"rewards_median_minmax.pdf")
    # display(plt_mean_minmax)

    # Plots.tex(plt, "./figures/"*name)
    if display_plot
        gui()
    end
    plt_quant, rollmean(stats["median"], 2), rollmean.((abs.(stats["median"] .- stats["quants"][2]), abs.(stats["median"] .- stats["quants"][1])), 2)
end

function eigen_and_rewards(paths_eigen::Vector{String}, paths_reward::Vector{String}; keep_rewards = false)
    l = length(Matrix(open_csv(paths_reward[1])))
    m = min([minimum(Matrix(open_csv(path))) for path in paths_reward]...)
    anim = @animate for (path,path_reward) in zip(paths_eigen,paths_reward)
        plt = plot(xlims=(1,l), ylims=(1.01*m, 1))
        plot!(plt[1], Matrix(open_csv(path_reward)), label="", xlabel="Episode", ylabel="Total reward")
        plot!(plt, YK_eigen_scatter(open_csv(path, types=ComplexF64)))
        end
end

function dict_trajectory(P)
    # gr()
    input = randn(MersenneTwister(0), 200)'
    # input = ones(100)'
    output = vec(lsim(P, input).y)
    plt = plot(lw=0.5, showaxis=false, legend=false, xticks=[], yticks=[], background_color=:transparent)
    for i in 1:10
        x = 2*i:2*i+100
        plot!(output[x] .- 0.015*i, label="", lw=2, lc=cgrad(:Blues), line_z=x)
    end
    # annotate!(0,0,text('{', :left, 100))
    savefig("./figures/"*"trajectory_set.png")
    display(plt)
end

## plotting PID experiments

function scatter_PID_params(path::String; path2::String="", file::String="PID_param_project.csv", types=Float64, display_plot=true, P=tf([-1, 1],[1, 3, 3, 1]), stop=0.01, scatter=true)
    # pgfplotsx()
    gr()
    direc = split(path,"/")[end]
    colors = palette(:default)
    xmin, xmax, ymin, ymax = -1.1, 2.1, -1.0, 0.70


    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    data_all = hcat(data...)


    plt = plot(annotations=[(-0.5, 0.5, "Unstable"), (1.5, -0.1, "Stable")], 
        xlims=(xmin, xmax), ylims=(ymin, ymax),
        )
    xlabel!(plt, L"k_p")
    ylabel!(plt, L"k_i")
    
    if !isempty(path2)
        paths2 = batch_paths(path2, "PID_param.csv")
        data2 = batch_data(paths2, types=types)
        data_all2 = hcat(data2...)


        datadist = [norm(col .- col2)>0.01 for (col, col2) in zip(eachcol(data_all), eachcol(data_all2))]
        # println(size(data_all2[:, datadist]))
        # println(size(data_all2))
        data_all = data_all[:, datadist]
        data_all2 = data_all2[:, datadist]
        plot!(plt, data_all[1,:], data_all[2,:], seriestype=:scatter, label="", markersize=2.0, markershape=:circle)
        plot!(plt, data_all2[1,:], data_all2[2,:], seriestype=:scatter, label="", markersize=1.0, markershape=:x)
    else
        plot!(plt, data_all[1,:], data_all[2,:], seriestype=:scatter, label="", markersize=2.0, markershape=:circle)
    end


    kp, ki, f1 = stabregionPID(tf([-1, 1],[1, 3, 3, 1]), exp10.(range(-2.0, stop=stop, length=1000)); doplot = false, form=:parallel)
    plot!(plt, kp, ki, label="", lw=2, color=:gray)

    # plot!(annotations=(-0.5, 0.5, "Unstable"))
    # plot!(annotations=(1.5, -0.1, "Stable"))

    plt

end


function mixture_PID_params(path::String; path2::String="", file::String="PID_param_project.csv", types=Float64, display_plot=true, P=tf([-1, 1],[1, 3, 3, 1]), stop=0.01)
    # pgfplotsx()
    gr()
    xmin, xmax, ymin, ymax = -1.1, 2.1, -0.25, 0.70
    plt = plot()
    direc = split(path,"/")[end]
    colors = palette(:default)

    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    data_all = hcat(data...)


    if !isempty(path2)
        paths2 = batch_paths(path2, "PID_param.csv")
        data2 = batch_data(paths2, types=types)
        data_all2 = hcat(data2...)
        

        datadist = [norm(col .- col2)>0.1 for (col, col2) in zip(eachcol(data_all), eachcol(data_all2))]
        # println(size(data_all2[:, datadist]))
        # println(size(data_all2))
        # data_all = data_all[:, datadist]
        data_all2 = data_all2[:, datadist]


        # println(datadist)
        mixture = [MvNormal([k[1], k[2]], 0.003*I) for k in eachcol(data_all)]
        mixture2 = [MvNormal([k[1], k[2]], 0.001*I) for k in eachcol(data_all2)]

        # Zplot = min.(0, sign.(Zmat .- Zmat2)).*log1p.(Zmat2) .+ log1p.(Zmat)
        mix1(x,y) = log1p.(sum([pdf(mix ,[x,y]) for mix in mixture]))
        mix2(x,y) = log1p.(sum([pdf(mix ,[x,y]) for mix in mixture2]))
        f(x,y) = mix1(x,y) - mix2(x,y)
        # f(x,y) = pdf(mixture ,[x,y]) - pdf(mixture2 ,[x,y])

        # g(x,y) = max(0.1, abs(f(x,y)))*sign(f(x,y))
        g(x,y) = f(x,y)
        # g(x,y) = mix1(x,y) > mix2(x,y) + 0.1 ? mix1(x,y) : -mix2(x,y)

        contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> g(x,y), c=cgrad(trueredbluewhite, [0, 0.65, 1]), colorbar=true, categorical=false, color = :black)    
        # contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> min(g(x,y), 0), c=cgrad(:tableau_red_blue_white), colorbar=true)    
        # Makie.tightlimits!(ax)


    else
        mixture = [ MvNormal([k[1], k[2]], 0.001*I) for k in eachcol(data_all)]
        contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> log1p.(sum([pdf(mix ,[x,y]) for mix in mixture])),  c=cgrad(trueBlues), colorbar=false)    
    end
    # mixture = MixtureModel([ MvNormal([k[1], k[2]], 0.025*I) for k in eachcol(data_all)] )
    # contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> pdf(mixture ,[x,y]),  c=cgrad(trueBlues), colorbar=false)

    # plt = plot(data_all[1,:], data_all[2,:], seriestype=:scatter)

    kp, ki, f1 = stabregionPID(P, exp10.(range(-2, stop=0.01, length=500)); doplot = false, form=:parallel)

    plot!(plt, kp, ki)
    plt

end

function heatmap_PID_params(path::String; path2::String="", file::String="PID_param_project.csv", types=Float64, display_plot=true, P=tf([-1, 1],[1, 3, 3, 1]), stop=0.01)
    # pgfplotsx()
    gr()
    direc = split(path,"/")[end]
    colors = palette(:default)
    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    data_all = hcat(data...)



    xmin, xmax, ymin, ymax = -1.1, 2.1, -0.25, 0.70
    δ = 0.015
    X = xmin:0.005:xmax
    Y = ymin:0.005:ymax
    Z = [sum([norm(xy .- col)<δ for col in eachcol(data_all)]) for xy in vec(Iterators.product(X,Y)|> collect)]
    Zmat = reshape(Z, (length(X), length(Y)))'

    plt = plot()

    if !isempty(path2)
        paths2 = batch_paths(path2, "PID_param.csv")
        data2 = batch_data(paths2, types=types)
        data_all2 = hcat(data2...)
        Z2 = [sum([norm(xy .- col)<δ for col in eachcol(data_all2)]) for xy in vec(Iterators.product(X,Y)|> collect)]
        Zmat2 = reshape(Z2, (length(X), length(Y)))'
        # Zplot = log1p.(Zmat) .- log1p.(Zmat2)
        # Zplot = Zmat .- Zmat2
        Zplot = min.(0, sign.(Zmat .- Zmat2)).*log1p.(Zmat2) .+ log1p.(Zmat)
        # Zplot = log1p.(Zplot)

        # Zplot = imfilter(Zplot, Kernel.gaussian(2))

        # println((Zmat2 .+ Zmat)/2)

        maxval = maximum(abs, Zplot)
    
        heatmap!(plt, X, Y, Zplot, c = cgrad([reverse(ColorSchemes.Reds.colors); colorant"white"; ColorSchemes.Blues.colors]), clims=(-maxval, maxval))

    else
        heatmap!(plt, X, Y, log1p.(Zmat), c=cgrad(trueBlues))
    end



    # plt = plot(data_all[1,:], data_all[2,:], seriestype=:scatter)



    kp, ki, f1 = stabregionPID(P, exp10.(range(-2, stop=0.01, length=500)); doplot = false, form=:parallel)

    plot!(plt, kp, ki)
    plt

end


function hexbin_PID_params(path::String; file::String="PID_param_project.csv", altfile::String="PID_param.csv", types=Float64, display_plot=true, P=tf([-1, 1],[1, 3, 3, 1]), stop=0.01)
    pyplot()
    xmin, xmax, ymin, ymax = -1.1, 2.1, -0.25, 0.70
    plt = plot()
    direc = split(path,"/")[end]
    colors = palette(:default)

    paths = batch_paths(path, file)
    data = batch_data(paths, types=types)
    data_all = hcat(data...)

    plt = plot(annotations=[(-0.5, 0.5, "Unstable"), (1.5, -0.1, "Stable")], 
        xlims=(xmin, xmax), ylims=(ymin, ymax),
        )
    # xlabel!(plt, L"k_p")
    # ylabel!(plt, L"k_i")


    if !isempty(altfile)
        paths2 = batch_paths(path, altfile)
        data2 = batch_data(paths2, types=types)
        data_all2 = hcat(data2...)

        
        datadist = [norm(col .- col2)>0.01 for (col, col2) in zip(eachcol(data_all), eachcol(data_all2))]
        # println(size(data_all2[:, datadist]))
        # println(size(data_all2))
        data_all = data_all[:, datadist]
        data_all2 = data_all2[:, datadist]
        # δ = 0.1
        # Z2 = [sum([norm(xy .- col)<δ for col in eachcol(data_all2)]) for xy in vec(Iterators.product(data_all2[1,:],data_all2[2,:])|> collect)]
        # Zmat2 = reshape(Z2, (length(data_all2[1,:]), length(data_all2[2,:])))

        # Zplot = min.(0, sign.(Zmat .- Zmat2)).*log1p.(Zmat2) .+ log1p.(Zmat)
        # mix1(x,y) = log1p.(sum([pdf(mix ,[x,y]) for mix in mixture]))
        # mix2(x,y) = log1p.(sum([pdf(mix ,[x,y]) for mix in mixture2]))
        # f(x,y) = mix1(x,y) - mix2(x,y)
        # f(x,y) = pdf(mixture ,[x,y]) - pdf(mixture2 ,[x,y])

        # g(x,y) = max(0.1, abs(f(x,y)))*sign(f(x,y))
        # g(x,y) = f(x,y)
        # g(x,y) = mix1(x,y) > mix2(x,y) + 0.1 ? mix1(x,y) : -mix2(x,y)

        # DATA = hcat(data_all2, data_all)
        hexbin!(plt, data_all2[1,:], data_all2[2,:], c=cgrad(:Reds), bins=25, linewidths=0.5, mincnt=1, alpha=0.5)
        hexbin!(plt, data_all[1,:], data_all[2,:], C=-1, c=cgrad(:Blues), bins=10, linewidths=0.5, mincnt=1)
    else
        mixture = [ MvNormal([k[1], k[2]], 0.001*I) for k in eachcol(data_all)]
        contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> log1p.(sum([pdf(mix ,[x,y]) for mix in mixture])),  c=cgrad(trueBlues), colorbar=false)    
    end
    # mixture = MixtureModel([ MvNormal([k[1], k[2]], 0.025*I) for k in eachcol(data_all)] )
    # contourf!(xmin:0.01:xmax, ymin:0.01:ymax, (x,y) -> pdf(mixture ,[x,y]),  c=cgrad(trueBlues), colorbar=false)

    # plt = plot(data_all[1,:], data_all[2,:], seriestype=:scatter)

    kp, ki, f1 = stabregionPID(P, exp10.(range(-2, stop=0.01, length=500)); doplot = false, form=:parallel)

    CSV.write(path*"/"*file, DataFrame(data_all, :auto))
    CSV.write(path*"/"*altfile, DataFrame(data_all2, :auto))
    CSV.write(path*"/"*"boundary.csv", DataFrame(hcat(kp, ki), :auto))


    plot!(plt, kp, ki)
    plt

end
