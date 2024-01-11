## see https://arxiv.org/abs/1802.08033
# https://discourse.julialang.org/t/random-orthogonal-matrices/9779/6

import Plots.plot
import Plots.plot!
using LinearAlgebra
# using RandomMatrices
using Flux
using Distributions

posdefMat(A) = A*A' + 0.1*I
# posdefMat(A) = 0.5*(softmax(A) + softmax(A')) + I
skewMat(A) = A - A'
orthoMat(A) = qr(A).Q

# # A = UpperTriangular(randn(5,5))
# A = randn(5,5)
# # M = (softmax(A) + softmax(A)')/4
# M = (softmax(A)*softmax(A)')/2
# M[diagind(M)] .= 1/2
# eigvals(M)
function Bmat(n)

    A = UpperTriangular(rand(Uniform(-10,10), n,n))
    # A = UpperTriangular(rand(Normal(0,1.0), n,n))

    scaleMat = I + ones(n,n)/(n-1) - I/(n-1)
    # A = randn(n,n)
    # M = (softmax(A) + softmax(A)')/4
    # M[diagind(M)] .= 1/2
    M = (softmax(A) + softmax(A)' + I)/4
    # M = (tanh(A+A') .* scaleMat + 2I )/4
    # M = ((softmax(A+I) + softmax(A+I)'))/2
    return M
    
end

# getSUB(n) = (posdefMat(randn(n,n)), exp(skewMat(randn(n,n))), diagm(rand(n,)))
# getSUB(n) = (posdefMat(randn(n,n)), qr(randn(n,n)).Q, diagm(rand(n,)))
# getSUB(n) = (posdefMat(randn(n,n)), rand(Haar(1),n), diagm(rand(n,)))
# getSUB(n) = (posdefMat(randn(n,n)), rand(Haar(1),n), Bmat(n))
getSUB(n; svd::Bool=false) = (posdefMat(randn(n,n)), qr(randn(n,n)).Q, Bmat(n))
getSUB(n; svd::Bool=true) = (posdefMat(randn(n,n)), qr(randn(n,n)).Q, diagm(sigmoid.(randn(n,))), qr(randn(n,n)).Q)



# getSUB(n) = (posdefMat(randn(n,n)), rand(Haar(1),n), (rand(Haar(1),n) + rand(Haar(1),n) + I)/3)

stableMat(S,U,B) = S\U*B*S

# stableMat(S,U,P,V) = pinv(S)U*P*V*S

function stableMat(S,U,P,V)
    # S = posdefMat(S)
    # U = svd(U).U
    # P = Diagonal(tanh.(exp.(P)))
    # V = svd(V).V
    # S, U, P, V = posdefMat(S), orthoMat(U), Diagonal(sigmoid.(P)), orthoMat(V)
    # S, U, P, V = posdefMat(S), svd(U).U, Diagonal(sigmoid.(P)), svd(V).V
    # U, P, V = svd(U).U, Diagonal(tanh.(svd(U).S)), svd(U).V
    # U*P*V'
    inv(S)*svd(U).U*Diagonal(tanh.(svd(U).S))*svd(U).V'*S
    # svd(U).U*Diagonal(tanh.(svd(U).S))*svd(U).V'
    # U*P*V'
    # U*P
end

function stableMat(S, M)
    # S, U, P, V = posdefMat(S), svd(M).U, Diagonal(0.5*(tanh.(svd(M).S) .+ 1.0)), svd(M).V

    S, U, P, V = posdefMat(S), svd(M).U, Diagonal(tanh.(svd(M).S)), svd(M).V
    S\U*P*V'*S
    # U*P*V'
    # orthoMat(S)*tanh.(Diagonal(M))
    # qr(S).Q*tanh.(Diagonal(M))
end
g(x) = 1.0 - softplus(-x + log(exp(1.0)-1.0))
function stableMat(M)
    # svd(M).U*Diagonal(clamp.(svd(M).S, 0.0, 1.0))*svd(M).V'
    # svd(M).U*Diagonal(g.(svd(M).S))*svd(M).V'
    svd(M).U*Diagonal(tanh.(svd(M).S))*svd(M).V'
    # Diagonal(tanh.(svd(M).S))*svd(M).V'
    # qr(M).Q*tanh.(Diagonal(qr(M).R))
    # M
end

getTraj(A,x,k) = [A^i*x for i in 0:k]

# softmax(v) = exp.(v) ./ sum(exp.(v))
function softmax(X::AbstractVecOrMat{T}, d::Integer=2)::AbstractVecOrMat where T<:AbstractFloat
    exp.(X) ./ sum(exp.(X), dims=d)
end
