using ControlSystems
using LinearAlgebra
using Statistics
using Plots
using LaTeXStrings
using RollingFunctions
# using SingularSpectrumAnalysis
# using OrdinaryDiffEq
# using NLopt, ForwardDiff

include("policies/stableMatrices.jl")

# persistently exciting input data (simple)
function excite(L::Int; num_data::Int=100, m::Int=1)
    num_data = max((m+1)*L , num_data)
    u = randn(num_data)
end

# Construct Hankel matrix
function Hankel(u::Vector, L::Int)
    N = length(u)
    H = vcat([u[i:N-L+i]' for i in 1:L]...)
end

## we can turn the above process into a function for ease
function complete_trajectory(u, y_init, H_u, H_y, H_y_init; solver="back")

    # if solver=="qr"
    #     α = qr([H_u; H_y_init]) \ [u; y_init]
    # elseif solve=="pinv"
    #     α = pinv([H_u; H_y_init]) * [u; y_init]
    # else
    #     α = [H_u; H_y_init] \ [u; y_init]

    α = Hankel_solver([H_u; H_y_init], [u; y_init], solver=solver)
    y = H_y*α

end

function Hankel_solver(A, b; solver="back")
    if solver=="qr"
        α = qr(A'*A) \ A'*b
    elseif solver=="pinv"
        α = qr(A'*A, Val(true)) \ A'*b
        # α = pinv(A) * b
    else
        α = A \ b
    end
    α
end

function closed_loop_trajectory(C, steps, H_u, H_y, H_y_init; λ::Float64=0, r="ones", solver="back")
    
    U = Any[]
    Y = Any[]
    x_c = zeros(size(C.B))
    ν = size(H_y_init, 1)
    u_traj = zeros(ν)
    y_traj = zeros(ν)
    u = 0
    if r == "ones"
        ref = ones(steps)
    else
        ref = sin.(1:steps)
    end

    for i in 1:steps
    
    
        # update the input trajectory
        append!(u_traj, u)
        popfirst!(u_traj)
        if λ!=0.0
            y_traj = complete_trajectory(u_traj, y_traj, H_u, H_y, H_y_init, λ::Float64, solver=solver)
        else
            y_traj = complete_trajectory(u_traj, y_traj, H_u, H_y, H_y_init, solver=solver)
        end
        popfirst!(y_traj)
        
        y = y_traj[end]
        e = ref[i] - y
        u = dot(C.C,x_c) + dot(C.D,e)
        x_c = dot(C.A,x_c) + dot(C.B,e)
        
        append!(U,u)
        append!(Y,y)
    end

    U, Y
end

# Add method to complete_trajectory
function complete_trajectory(u, y_init, H_u, H_y, H_y_init, λ::Float64; solver="back")
    α = Hankel_solver([H_u; H_y_init; λ*I(size(H_u,2))], [u; y_init; zeros(size(H_u,2))], solver = solver)
    y = H_y*α
end

function complete_trajectory(u, y_init, H_y, A)
    α = A * [u; y_init]
    y = H_y*α
end

function noisy_trajectory(P::StateSpace{ControlSystems.Discrete{T}}; order::Int=10, num_data::Int=100, σ²::Float64=0.01) where T

    u = excite(order, num_data=num_data)
    x = zeros(size(P.B))
    y = Float64[]
    
    for i in 1:length(u)
        
        append!(y,dot(P.C,x) + dot(P.D,u[i]) + σ²*randn(Float64))
        x = P.A*x + P.B*u[i]
        
    end

    return vec(u), vec(y)
end

# add a 'minimal' excitation method
function excite(L::Int, impulse::Int; num_data::Int=100, m::Int=1)
    u = vec([zeros(L-1); ones(impulse); zeros(L)])
end

function get_rand_trajectory(H_u, H_y, H_y_init, steps; λ::Float64=0.0, solver="back", input="rand")

    L = size(H_u,1)
    ν = size(H_y_init,1)

    if input == "step"
        U = [zeros(ν); ones(steps)]
    else
        U = [zeros(ν); randn(steps)]
    end
    Y = zeros(ν)

    while length(Y)+L-ν <= steps

        if λ!=0.0
            y_complete = complete_trajectory(U[length(Y)-ν+1:length(Y)+L-ν], Y[end-ν+1:end], H_u, H_y, H_y_init, λ, solver=solver)
        else
            y_complete = complete_trajectory(U[length(Y)-ν+1:length(Y)+L-ν], Y[end-ν+1:end], H_u, H_y, H_y_init, solver=solver)
        end
        if L-ν==0.0
            append!(Y, y_complete[end])
        else
            append!(Y, y_complete[ν+1:end])
        end
    end

    U = U[1:length(Y)]
    return U, Y
end


function get_rand_trajectory(u, y, L::Int, steps::Int, P, σ²; K::Int = 1, λ::Float64=0.0, solver="back")

    y_noisy = y .+ σ²*randn(size(y))
        
    u_train_smooth = √(K)*rollmean(vec(u), K)
    y_train_smooth = √(K)*rollmean(vec(y_noisy), K)
    
    H_u = Hankel(u_train_smooth, L)
    H_y = Hankel(vec(y_train_smooth), L)
    H_y_nu = Hankel(vec(y_train_smooth)[1:end - 1], L-1)

    if λ != 0.0
        ū, ȳ = get_rand_trajectory(H_u, H_y, H_y_nu, steps, λ=λ, solver=solver)
    else
        ū, ȳ = get_rand_trajectory(H_u, H_y, H_y_nu, steps, solver=solver)
    end

    y_true, t, x, = lsim(P, ū')
    y_true = vec(y_true)

    error = mean(abs.(y_true - vec(ȳ)))

    # return cumulative error across rollouts as well as the data for the last rollout
    return ū, ȳ, y_true, t, error
    
end

function complete_prediction(u, y_init, H_u, H_y, H_y_nu)

    α = [H_u; H_y_nu]\[u; y_init]
    y = H_y[end:end,:]*α

end

function YK_trajectory(Q, H_u, H_y, H_y_init, input)

    Y = Any[]
    u_traj = zeros(size(H_u,1))
    y_traj = zeros(size(H_y_init,1))
    u = 0
    y = 0
    x_c = zeros(size(Q.B))

    for r in input
    
        # update the input trajectory
        append!(u_traj, u)
        popfirst!(u_traj)

        y = complete_prediction(u_traj, y_traj, H_u, H_y, H_y_init)
        append!(y_traj, y)
        popfirst!(y_traj)
        
        u = dot(Q.C,x_c) + dot(Q.D,r)
        x_c = vec(Q.A*x_c) .+ vec(Q.B*r)
        
    end

    return sum(x_c)

end



"""
The rest of this is a bunch of random code from earlier experiments that didn't make it into the paper
"""

p_A(p) = stableMat(reshape(p[1:ν^2], (ν, ν)))

p_B(p) = reshape(p[ν^2 + 1: ν^2 + ν], (ν, 1))
p_C(p) = reshape(p[ν^2 + ν + 1: ν^2 + 2ν], (1, ν))
p_D(p) = reshape(p[end:end], (1,1))

Q_param(p)     = ss(p_A(p), p_B(p), p_C(p), p_D(p), h)

struct Traj
    u::Vector
    y::Vector
    ν::Int64
end

struct Traj_full
    u::Vector
    y::Vector
    ν::Int64
end


struct OptTraj
    c::Function
    f::Function
    ν::Int
end




function (traj_Q::Traj)(Q, Pd)

    ν = traj_Q.ν
    L = ν + 1
    u = traj_Q.u
    y = traj_Q.y

    H_u = Hankel(vec(u), L)
    H_y = Hankel(vec(y), L)

    H_u_init = Hankel(vec(u)[1:length(u) - L + ν], ν)
    H_y_init = Hankel(vec(y)[1:length(y) - L + ν], ν)

    # Initialize the rollout 
    u_traj = lsim(Q, [zeros(ν); ones(250)]').y'
    y_traj = lsim(Pd, u_traj').y'[1:size(H_y_init,1)]
    H_pinv = pinv([H_u_init; H_y_init])

    # Get the rest of the rollout
    ŷ = 0
    for (i,u) in enumerate(u_traj[size(H_u_init,1)+1:end])

        ŷ = dot(H_y[end:end,:], H_pinv * vcat(u_traj[i+1:i+size(H_u_init,1)], y_traj[end-size(H_y_init,1)+1:end]))

        append!(y_traj,ŷ)
    end

    y_traj[ν: end]

end

# timedomain(p) = traj_Q(Q_param(p), Pd)

function constraintfun(p)

    svd(p_A(p)).S .- 1.0
    
end

# g_cfg = ForwardDiff.JacobianConfig(constraintfun, p)

function c(result, p::Vector, grad)
    if length(grad) > 0
        grad .= ForwardDiff.jacobian(constraintfun,p)'
    end
    result .= constraintfun(p)
end

function costfun_eval(y)
    # y = timedomain(p)
    # sum(abs.(1 .- time_domain(p)))
    push!(iter_counter, 1.0)
    mean(abs.(1 .- y))
    # mean(abs.(1 .- timedomain(p)))
    # mean((1 .- time_domain(p)).^2)
    # mean(abs,1 .- y) # ~ Integrated absolute error IAE
end

function costfun(y)
    push!(iter_counter, 1.0)

    gamma = [0.99^i for i in 0:length(y)-1]
    mean(gamma.*abs.(1 .- y))
    # mean(abs.(1 .- y))
end


# f_cfg = ForwardDiff.GradientConfig(costfun, p)

# function f(p::Vector, grad::Matrix)
#     if length(grad) > 0
#         grad .= ForwardDiff.gradient(costfun,p)
#         # grad .= gradient(p  -> costfun(p),p)
#     end
#     costfun(p)
# end

function f(p::Vector, grad::Vector=[])
    if length(grad) > 0
        grad .= ForwardDiff.gradient(costfun,p)
    end
    costfun(p)
end

function f(costfun::Function, p::Vector, grad::Vector=[])
    if length(grad) > 0
        grad .= ForwardDiff.gradient(costfun,p)
    end
    costfun(p)
end




function (traj_full::Traj_full)(Q, Pd; integral = true)

    ν = traj_full.ν
    L = ν + 1
    u = traj_full.u
    y = traj_full.y

    H_u = Hankel(vec(u), L)
    H_y = Hankel(vec(y), L)

    H_u_init = Hankel(vec(u)[1:length(u) - L + ν], ν)
    H_y_init = Hankel(vec(y)[1:length(y) - L + ν], ν)

    # Initialize the rollout 
    ref = [zeros(ν); ones(300)]'
    u_traj = lsim(Q, ref).y'[1:L,1]
    state_Q = lsim(Q, ref).x'[L, :]
    # println(size(lsim(Q, ref).x))
    y_traj = lsim(Pd, u_traj').y'[1:L]
    state_P = lsim(Pd, u_traj').x'[L, :]
    outputs = Any[]
    outputs_H = Any[]

    # Get the rest of the rollout
    # y = y_traj[end]
    u = u_traj[end]

    H_pinv = pinv([H_u_init; H_y_init])

    # state_P = zeros(size(Pd.A,1), 1)
    # state_Q = zeros(size(Q.A,1), 1)
    int_error = 0
    for (i,r) in enumerate(ref[size(H_y_init,1)+1:end])

        state_P = vec(Pd.A*state_P) + vec(Pd.B*u)
        y = dot(Pd.C,state_P) + 0.01*randn()

        e = r .- y

        # y_H = dot(H_y[end:end,:], [H_u_init; H_y_init; 1.0*I(size(H_u_init,2))] \ vcat(u_traj[end-size(H_u_init,1)+1:end], y_traj[end-size(H_y_init,1)+1:end], zeros(size(H_u_init,2))))
        # y_H = dot(H_y[end:end,:], [H_u_init; H_y_init] \ vcat(u_traj[end-size(H_u_init,1)+1:end], y_traj[end-size(H_y_init,1)+1:end]))
        y_H = dot(H_y[end:end,:], H_pinv * vcat(u_traj[end-size(H_u_init,1)+1:end], y_traj[end-size(H_y_init,1)+1:end]))

        input = e .+ y_H

        u = dot(Q.C, state_Q) + dot(Q.D, input)
        state_Q = vec(Q.A*state_Q) + vec(Q.B*input)

        if integral

            int_error += h*e

            u_int = 0.05*int_error

            u += u_int
            
        end

        append!(u_traj, u)
        # append!(y_traj,y_H)
        append!(y_traj,y)

        append!(outputs, y)
        append!(outputs_H, y_H)

    end

    outputs_H, outputs
    
end

# timedomain_RL(p) = timedomain_RL(Q_param(p), Pd)[2]

function costfun_RL(y)
    # y = timedomain(p)
    # sum(abs.(1 .- timedomain(p)))
    mean(abs.(1 .- y))
    # mean(abs.(1 .- timedomain_RL(p)))
    # mean((1 .- timedomain(p)).^2)
    # mean(abs,1 .- y) # ~ Integrated absolute error IAE
end

# f_cfg = ForwardDiff.GradientConfig(costfun, p)

function f_RL(p::Vector, grad::Vector=[])
    if length(grad) > 0
        grad .= ForwardDiff.gradient(costfun_RL,p)
        # grad .= gradient(p  -> costfun(p),p)
    end
    costfun_RL(p)
end

function f_RL(p::Vector, grad::Vector=[])
    if length(grad) > 0
        grad .= ForwardDiff.gradient(f_RL.costfun_RL,p)
        # grad .= gradient(p  -> costfun(p),p)
    end
    f_RL.costfun_RL(p)
end


# function runopt(p; f_tol = 1e-5, x_tol = 1e-3, c_tol=1e-8)
#     # opt = Opt(:LD_MMA, d)
#     opt = Opt(:LD_SLSQP, d)
#     # opt.lower_bounds = 1e-6ones(d)
#     opt.xtol_rel = x_tol
#     opt.ftol_rel = f_tol
#     opt.min_objective = f
#     # inequality_constraint!(opt, c, c_tol*ones(1))
#     NLopt.optimize(opt, p)
# end
global iter_counter=[]

function (runopt::OptTraj)(p; f_tol = 1e-5, x_tol = 1e-3, c_tol=1e-8, mode="static")
    # opt = Opt(:LD_MMA, d)
    opt = Opt(:LD_SLSQP, length(p))
    # opt.lower_bounds = 1e-6ones(d)
    opt.xtol_rel = x_tol
    opt.ftol_rel = f_tol
    # mode == "static" ? f =  : f = f_RL(p)
    opt.min_objective = runopt.f
    opt.maxtime = 20
    # inequality_constraint!(opt, runopt.c)
    # inequality_constraint!(opt, runopt.c, c_tol*ones(runopt.ν))
    NLopt.optimize(opt, p), iter_counter
end

# function runopt_RL(p; f_tol = 1e-5, x_tol = 1e-3, c_tol=1e-8)
#     # opt = Opt(:LD_MMA, d)
#     opt = Opt(:LD_SLSQP, d)
#     # opt.lower_bounds = 1e-6ones(d)
#     opt.xtol_rel = x_tol
#     opt.ftol_rel = f_tol
#     opt.min_objective = f_RL
#     inequality_constraint!(opt, c, c_tol*ones(ν))
#     NLopt.optimize(opt, p)
# end

# function runopt(p, u, y, ν, P; f_tol = 1e-5, x_tol = 1e-3, c_tol=1e-8, mode="static")


#     H_u = Hankel(vec(u), L)
#     H_y = Hankel(vec(y), L)
#     H_u_nu = Hankel(vec(u)[1:length(u) - L + ν], ν)
#     H_y_nu = Hankel(vec(y)[1:length(y) - L + ν], ν)

#     timedomain(p) = traj_Q(Q_param(p), H_u_nu, H_y, H_y_nu, Pd)

#     return runopt(p; f_tol = f_tol, x_tol = x_tol, c_tol=c_tol, mode=mode)
# end



# function (runopt::OptTraj)(p; f_tol = 1e-5, x_tol = 1e-3, c_tol=1e-8, mode="static")
#     # timedomain = runopt.timedomain
#     # constraintfun = runopt.constraintfun
#     # c = runopt.c
#     # cost = runopt.cost
#     # f = runopt.f
#     # runopt = runopt.runopt

#     runopt.runopt(p)

# end
