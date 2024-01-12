using DataFrames
using CSV
using Glob
using Statistics
using StatsBase

function open_csv(path; types=Float64)
    DataFrame(CSV.File(path, types=types))
end

function batch_paths(path::String, file::String)
    sort(glob("**/"*file, path))
end

function batch_data(paths::Vector{String}; types=Float64)
    [Matrix(open_csv(path, types=types)) for path in paths]
end

function stack_batch(batch_data::Vector)
    vcat([vec(data) for data in batch_data]...)
end



function batch_stats(batch_data::Vector)

    # batch_data = batch_data(paths, types=types)
    # println([size(data) for data in batch_data])
    # println(open_csv(paths[2], types=types)[!,25:30])
    # plt = plot()
    # plot!(plt, batch_data, label="")
    # display(plt)    
    # vectorize_batch = [vec(data) for data in batch_data]
    group_data = eachrow(hcat([vec(data) for data in batch_data]...))

    # group_data = [vec(data) for data in batch_data]

    # min_batch = min.([vec(data) for data in batch_data]...)
    # max_batch = max.([vec(data) for data in batch_data]...)
    # reshape(max_batch, (size(batch_data[1])))
    p = 0.75
    Dict("mean"=>mean.(group_data), "std"=>std.(group_data),
        "median"=>median.(group_data), "iqr"=>iqr.(group_data)./2,
        "min"=>minimum.(group_data), "max"=>maximum.(group_data),
        "quant-range"=>(quantile.(group_data, p) - quantile.(group_data, 1.0-p))./2,
        "quants"=>(quantile.(group_data, p), quantile.(group_data, 1.0-p))
        )

end

function knn_var(data::Vector; k=4)

    # batch_data = batch_data(paths, types=types)
    # batch_data_merge = vcat([vec(mat) for mat in batch_data]...)
    # println(batch_data_merge)
    knn_var = Array{Float64}(undef, size(data))
    for (i,coord) in enumerate(data)
        knn_var[i] = max(var(sort(data, by=x->norm(x .- coord))[2:k+1], corrected=true, mean=coord), 0.01)
    end
    knn_var
end

function get_mixture(data::Vector; var=nothing)
    if var === nothing
        MixtureModel( [ MvNormal([real(z), imag(z)], 0.025*I) for z in data] )
    else
        MixtureModel( [ MvNormal([real(z), imag(z)], std*I) for (z, std) in zip(data, var)] )
    end
end

function get_mixture(P::Vector{Distribution})
    MixtureModel(P)
end