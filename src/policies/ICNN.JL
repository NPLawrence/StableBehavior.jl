using LinearAlgebra
using Flux
using Flux.Losses
using Flux: @functor, glorot_uniform, kaiming_normal
export nonnegDense
"""
    ICNN

Partially input convex neural network [1].
# References
[1] B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in 34th International Conference on Machine Learning, ICML 2017, 2017, vol. 1, pp. 192–206.
# Variables
(!!NOTICE!!) The variables in this code probably follow the notation of the original paper as follows:
`x` and `y` denote non-convex and convex inputs, respectively.
`g̃` and `g` denote activation functions of `x`- and `y`- paths, respectively.
Furthermore, `u` and `z` denote `x`- and `y`-path variables, respectively.
"""

struct nonnegDense{F, M<:AbstractMatrix}
    weight::M
    # bias::B
    σ::F
    function nonnegDense(W::M, σ::F = identity) where {M<:AbstractMatrix, F}
    #   b = create_bias(W, bias, size(W,1))
      new{F,M}(W, σ)
    end
end
  
function nonnegDense(in::Integer,out::Integer, σ = identity;
                init = Flux.kaiming_uniform)
    nonnegDense(init(out, in), σ)
end
  
@functor nonnegDense
  
function (a::nonnegDense)(x::AbstractVecOrMat)
    return a.σ.(softplus.(a.weight) * x)
end
  
(a::nonnegDense)(x::AbstractArray) = 
reshape(a(reshape(x, size(x,1), :)), :, size(x)[2:end]...)
  
function Base.show(io::IO, l::nonnegDense)
    print(io, "nonnegDense(", size(l.weight, 2), " => ", size(l.weight, 1))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end


struct ICNN
    conv_layers::Vector
    reg_layers::Vector
    out_layer::Vector
    act::Function
end

@functor ICNN

function ICNN(n,m,l,act::Function=Flux.relu)
    conv_layers = [nonnegDense(m,m) for i in 1:l]
    reg_layers = [Dense(n,m) for i in 1:l+1]
    out_layer = [nonnegDense(m,1), Dense(n,1)]
    ICNN(conv_layers, reg_layers, out_layer, act)
end

function (model::ICNN)(x::AbstractArray)

    m = sqrt(size(model.conv_layers[1].weight,2))
    z = model.reg_layers[1](x)
    z = model.act(z)
    for (Wz, Wx) in zip(model.conv_layers, model.reg_layers[2:end])
        z = model.act.(Wz(z) ./ m .+ Wx(x))
    end
    model.out_layer[1](z) .+ model.out_layer[2](x)

end