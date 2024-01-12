## see https://arxiv.org/abs/1802.08033
# https://discourse.julialang.org/t/random-orthogonal-matrices/9779/6

import Plots.plot
import Plots.plot!
using LinearAlgebra

posdefMat(A) = A*A' + 0.1*I
skewMat(A) = A - A'
orthoMat(A) = qr(A).Q

getSUB(n; svd::Bool=true) = (posdefMat(randn(n,n)), qr(randn(n,n)).Q, diagm(sigmoid.(randn(n,))), qr(randn(n,n)).Q)

stableMat(S,U,B) = S\U*B*S

# stableMat(S,U,P,V) = pinv(S)U*P*V*S

function stableMat(S,U,P,V)
    inv(S)*svd(U).U*Diagonal(tanh.(svd(U).S))*svd(U).V'*S
end

function stableMat(S, M)
    S, U, P, V = posdefMat(S), svd(M).U, Diagonal(tanh.(svd(M).S)), svd(M).V
    S\U*P*V'*S
end

g(x) = 1.0 - softplus(-x + log(exp(1.0)-1.0))

function stableMat(M)
    svd(M).U*Diagonal(0.95*tanh.(svd(M).S))*svd(M).V'
end

getTraj(A,x,k) = [A^i*x for i in 0:k]

# softmax(v) = exp.(v) ./ sum(exp.(v))
function softmax(X::AbstractVecOrMat{T}, d::Integer=2)::AbstractVecOrMat where T<:AbstractFloat
    exp.(X) ./ sum(exp.(X), dims=d)
end
