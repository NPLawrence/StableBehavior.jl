export YK_param

using Flux
using Flux: @functor, glorot_uniform, kaiming_normal
using ControlSystems
using Flux: params


include("../Hankel.jl")
include("stableMatrices.jl")
# include("../qr_backprop_mwe.jl")

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

# struct YK_param_stable{F <: Function, M <: AbstractArray}
#     M::M
#     B::M
#     C::M
#     D::M
#     σ::F
# end

# YK_param(A, B, C, D) = YK_param(A, B, C, D, identity)
YK_param_stable(S, U, P, V, B, C, D) = YK_param_stable(S, U, P, V, B, C, D, identity)
# YK_param_stable(M, B, C, D) = YK_param_stable(M, B, C, D, identity)

@functor YK_param
@functor YK_param_stable

function YK_param(n::Integer, m::Integer, p::Integer, σ = identity; rng=MersenneTwister(123),
    initYK = kaiming_normal, stable = true)

    if stable
        # s = tf("s")
        # init = ss(c2d((10)/(100s+1), 0.5))
        # return YK_param_stable(init.A, init.B, init.C, init.D, σ)
        # return YK_param_stable(initYK(rng, n, n), initYK(rng, n, m), initYK(rng, p, n), initYK(rng, p, m), σ)
        return YK_param_stable(initYK(rng, n, n), initYK(rng, n, n), initYK(rng, n), initYK(rng, n, n), initYK(rng, n, m), initYK(rng, p, n), initYK(rng, p, m), σ)

        # return YK_param_stable(randn(n, n), randn(n, m), randn(p, n), randn(p,m), σ)
    else
        return YK_param(initYK(rng, n, n), initYK(rng, n, m), initYK(rng, p, n), initYK(rng, p, m), σ)
    end

    ## Troubleshooting with handpicked A,B,C,D terms
    # s = tf("s")
    # Q = ss(c2d((s + 0.5)*(s + 0.6) / ((s+0.6)*(s+0.9)),1))
    # A, B, C, D = Q.A, Q.B, Q.C, Q.D
    # return YK_param(A, B, C, D, σ)

end

function act(A,B,C,D,x)

    n, m = size(A,2), size(B,2)

    PE_in, PE_out = SS_signal(A, B, C, D)
    H_in = H_matrix(PE_in, n+1)
    H_out = H_matrix(PE_out[1:end-1], n)
    H_out_full = H_matrix(PE_out[2:end], n)

    Qinput = x[1:n+1,:]
    Qoutput = x[n+2:2(n+1)-1,:]
    u = *(H_out_full[end:end,:], pinv([H_in;H_out])*[Qinput; Qoutput])

    # α = [H_in; H_out]\[Qinput; Qoutput]
    # α = [H_in; H_out; 0.1*I(size(H_in,2))]\[Qinput; Qoutput; zeros(size(H_in,2), size(Qinput,2))]
    # y = H_y*alpha
    # println(typeof(H_y[end:end,:]*alpha)) 
    # println(typeof(α))
    # u = *(H_out_full[end:end,:],α)

    # println("Norm: ", norm([H_in; H_out]*α .- [Qinput; Qoutput]))
    # println("Action: ", u)
    

    # u = complete_prediction(Qinput, Qoutput, H_in, H_out_full, H_out)
    return u

end

function (model::YK_param)(x::AbstractArray)
    A, B, C, D, σ = model.A, model.B, model.C, model.D, model.σ
    
    u = act(A,B,C,D,x)
    
    return u
end

function (model::YK_param_stable)(x::AbstractArray)
    S, U, P, V, B, C, D, σ =  model.S, model.U, model.P, model.V, model.B, model.C, model.D, model.σ

    # M, B, C, D, σ =  model.M, model.B, model.C, model.D, model.σ

    # S, U, P, V,  = posdefMat(model.S), qr(model.U).Q, Diagonal(sigmoid.(model.P)), qr(model.V).Q

    ## Use a simple Householder matrix to make sure everything else works
    # S, P  = posdefMat(model.S), Diagonal(sigmoid.(model.P))
    # u = diag(sigmoid.(model.U)) / norm(diag(sigmoid.(model.U)))^2
    # v = diag(sigmoid.(model.V)) / norm(diag(sigmoid.(model.V)))^2
    # U, V = I - u*u', I - v*v'

    A = stableMat(S,U,P,V)
    # A = stableMat(M)

    u = act(A,B,C,D,x)

    # A = model.S
    # u = act(A,B,C,D,x)

    return u
end




# function (test::YK_param)(x::AbstractArray; ts=0.1)
#     # A, B, C, D, σ = model.A, model.B, model.C, model.D, model.σ
#     s = tf("s")
#     Q = ss(c2d((s + rand(1)[1]) / ((s + rand(1)[1])*(s + rand(1)[1])),ts))
#     A, B, C, D = Q.A, Q.B, Q.C, Q.D
#     n, m = size(A,2), size(B,2)
#     # qprev, input_prev, input = x[1:n], x[n+1:n+m], x[n+1+m:n+2*m]

#     # q = σ.(A*qprev .+ B*input_prev)
#     # u = σ.(C*q .+ D*input)

#     # return [q; u]
#     PE_in = vec([1; 1; 0; 0])
#     PE_out = vec([D; C*B + D; C*A*B + C*B; C*A^2*B + C*A*B])
#     H_in = H_matrix(PE_in, 2)
#     H_out = H_matrix(PE_out[1:end-1], 1)
#     H_out_full = H_matrix(PE_out, 2)

#     u = complete_prediction(x[end-1:end,:], x[end-2:end-2,:], H_in, H_out_full, H_out)

#     return u
# end

# n = 2
# m = 1
# p = 1

# test_model = Chain(
#     YK_param(n, m, p, identity)
#     )
# x = [1;2;3;4;5;6]
# f(x) = test_model(x)
# loss(x) = sum(f(x))

# grads = gradient(()->loss(x), Flux.params(test_model))

# println("Param: ", Flux.params(test_model)[1])
# println("Grad: ", grads[Flux.params(test_model)[1]])

# Xtrain = rand(3,2)
# println(Xtrain)

# array_loader = Flux.DataLoader(Xtrain, batchsize=2, shuffle=false)

# for x in array_loader
#     @assert size(x) == (3, 2)
#     println("x ", x)
#     println("val ", test_model(x))
#   end
