using ControlSystems
using LinearAlgebra
using Random
using Plots
using Printf
using GenericLinearAlgebra: svd
using NLopt, ForwardDiff

include("stableMatrices.jl")
include("../nbtools.jl")

p_A(p,ν) = stableMat(reshape(p[1:ν^2], (ν, ν)))
p_B(p,ν) = reshape(p[ν^2 + 1: ν^2 + ν], (ν, 1))
p_C(p,ν) = reshape(p[ν^2 + ν + 1: ν^2 + 2ν], (1, ν))
p_D(p,ν) = reshape(p[end:end], (1,1))
Q_param(p, h, ν)     = ss(p_A(p,ν), p_B(p,ν), p_C(p,ν), p_D(p,ν), h)

mutable struct sys_data
    sys::StateSpace{ControlSystems.Discrete{Float64}, Float64}
    u::Vector
    y::Vector
    ν::Int64
    h::Float64
    pinit::Vector
    H_y::Matrix
    H_pinv::Matrix
    ν_design::Int64
end

struct OptStable
    c::Function
    f::Function
end


function XYWZ_traj(p, sys::sys_data, u)
    ν = sys.ν_design
    W = Q_param(p[1:ν^2+2ν+1], sys.h, ν)
    Z = Q_param(p[ν^2+2ν+2:2ν^2+4ν+2], sys.h, ν)
    X = Y_discrete(1.0, 0.0, Ts=sys.h, τ=1.0)*Q_param(p[2ν^2+4ν+3:3ν^2+6ν+3], sys.h, ν)
    Y = -Y_discrete(p[end-1], p[end], Ts=sys.h, τ=1.0)*Q_param(p[2ν^2+4ν+3:3ν^2+6ν+3], sys.h, ν)

    X_data = lsim(X, u').y'
    Y_data = lsim(Y, u').y'
    W_data = lsim(W, u').y'
    Z_data = lsim(Z, u').y'

    X_G = lsim(X, sys.y[1:200]').y'
    Y_G = lsim(Y, sys.y[1:200]').y'

    return X_data, Y_data, W_data, Z_data, X_G, Y_G
end

function Y_discrete(kp, ki; Ts=1.0, τ=1.0)
    # z transform of (kp s + ki) / (τ s + 1)
    z = tf("z", Ts)
    c = exp(-Ts/τ)
    a = kp / τ

    (a*z - a + ki*(1-c)) / (z - c)
end

function Hankel_rollout(sys::sys_data, input)
    ν = sys.ν
    # L = ν + 1
    u = sys.u
    y = sys.y

    u_traj = input
    y_traj = lsim(sys.sys,u_traj').y'[1:ν] # this is to align the system trajectory with the inputs and for ForwardDiff to work

    for i in 1:200-ν
        y = dot(sys.H_y[end:end,:], sys.H_pinv * vcat(u_traj[i:i+ν-1], y_traj[end-ν+1:end]))
        append!(y_traj, y)
    end
    return y_traj
end


function (rollout::sys_data)(p)

    X_data, Y_data, W_data, Z_data, X_G, Y_G = XYWZ_traj(p, rollout, rollout.u[1:200])
    G_Y = Hankel_rollout(rollout, Y_data)
    G_Z = Hankel_rollout(rollout, Z_data)
    I_X = X_data
    I_W = W_data

    W_I = W_data
    Z_I = Z_data

    I_X .- G_Y, I_W .- G_Z, W_I .- X_G, Z_I .- Y_G, rollout.u
end

function cost_rollout(s1, s2, s3, s4, input)
    # mean(abs.(s1[1:200] .- input[1:200])) + mean(abs.(s2[1:200])) + mean(abs.(s3[1:200])) + mean(abs.(s4[1:200] .- input[1:200]))
    mean((s1[1:200] .- input[1:200]).^2) + mean(s2[1:200].^2) + mean(s3[1:200].^2) + mean((s4[1:200] .- input[1:200]).^2)
end

function f(costfun::Function, p::Vector, grad::Vector=[])
    if length(grad) > 0
        grad .= ForwardDiff.gradient(costfun,p)
    end
    costfun(p)
end

function (runopt::OptStable)(p; f_tol = 1e-3, x_tol = 1e-3, c_tol=1e-8, mode="constrained")
    # opt = Opt(:LD_MMA, length(p))
    # opt = Opt(:LD_CCSAQ, length(p))
    opt = Opt(:LD_SLSQP, length(p))
    # opt = Opt(:LD_LBFGS, length(p))
    # opt = Opt(:LN_COBYLA, length(p))
    opt.min_objective = runopt.f
    opt.maxtime = 120
    opt.xtol_rel = x_tol
    # opt.ftol_rel = f_tol
    # opt.ftol_abs = 0.5
    if mode == "constrained"
        inequality_constraint!(opt, runopt.c, c_tol)
    end

    NLopt.optimize(opt, p)
end

function project_sol(sys::sys_data, p_unstable; tol=0.05)
    # projectfun = p -> mean(abs.(p[end-1:end] .- p_unstable[end-1:end]))
    projectfun = p -> norm(p[end-1:end] .- p_unstable[end-1:end])
    # projectfun = p -> norm(p[end-1:end] .- p_unstable[end-1:end], 2)^2
    constraintfun = p -> cost_rollout(sys(p)...) - tol

    g(p::Vector, grad::Vector=[]) = f(projectfun, p, grad)
    h(p::Vector, grad::Vector=[]) = f(constraintfun, p, grad)
    
    OptStable(h, g)(p_unstable)
end

function log_barrier(sys::sys_data, p_unstable; tol=0.001)

    projectfun = p -> mean((p[end-1:end] .- p_unstable[end-1:end]).^2)

    logfun = p -> -log(tol - cost_rollout(sys(p)...))
    constraintfun = p -> cost_rollout(sys(p)...) - tol

    obj = p -> cost_rollout(sys(p)...) + 0.01projectfun(p)

    g(p::Vector, grad::Vector=[]) = f(obj, p, grad)

    OptStable(g, g)(randn(3sys.ν_design^2+6sys.ν_design+6), mode="")

end

function check_sol(sys::sys_data, p)
    
    # fstar, p, = OptStable(c, f)(p)

    s1, s2, s3, s4, input = sys(p)
    X_data, Y_data, W_data, Z_data, X_G, Y_G = XYWZ_traj(p, sys, input)
    G_Y_tf = lsim(sys.sys, Y_data').y'
    G_Z_tf = lsim(sys.sys, Z_data').y'
    
    s1_tf, s2_tf, s3_tf, s4_tf = X_data .- G_Y_tf, W_data .- G_Z_tf, W_data .- X_G, Z_data .- Y_G

    plt = plot( 
    xaxis  = ("Time", ),
    xlims = (0, 100),
    layout = (3,2)
)
    plot!(plt[1], s1 .- sys.u, linetype=:step)
    plot!(plt[1], s1_tf, linetype=:step, ls=:dash)
    plot!(plt[2], s2, linetype=:step)
    plot!(plt[2], s2_tf, linetype=:step, ls=:dash)
    plot!(plt[3], s3, linetype=:step)
    plot!(plt[3], s3_tf, linetype=:step, ls=:dash)
    plot!(plt[4], s4 .- sys.u, linetype=:step)
    plot!(plt[4], s4_tf, linetype=:step, ls=:dash)

    cl = feedback(pid(p[end-1],p[end],form=:parallel,Ts=h)*sys.sys)

    # cl = feedback(sys.sys, -pid(p[end-1],p[end],form=:parallel,Ts=sys.h))
    z = tf("z", sys.h)
    println(abs.(pole(1 / (1 + pid(p[end-1],p[end],form=:parallel,Ts=sys.h)*sys.sys))))
    println(abs.(pole(1 / (1 - pid(p[end-1],p[end],form=:parallel,Ts=sys.h)*sys.sys))))
    println("kp ", p[end-1])
    println("ki ", p[end])

    # println(pstar[end-2])

    println(cost(p))
    println(cost(sys.pinit))

    plot!(plt[5], step(cl))

    plt, p
end

# plt, pstar = check_sol(sys_data(P_d, pe_u, pe_y, ν, h, pinit), pinit)
# plt