import Flux
import Flux: NNlib
import Dates
import Memoize
import StatsBase
import Measurements
import BSON

#= ====================================== =#
#  structs/constructors for the TGLFmodel
#= ====================================== =#
# TGLFmodel abstract type, since we could have different models
abstract type TGLFmodel end

# TGLFNNmodel
struct TGLFNNmodel <: TGLFmodel
    fluxmodel::Flux.Chain
    name::String
    date::Dates.DateTime
    xnames::Vector{String}
    ynames::Vector{String}
    xm::Vector{Float32}
    xσ::Vector{Float32}
    ym::Vector{Float32}
    yσ::Vector{Float32}
    xbounds::Array{Float32}
    ybounds::Array{Float32}
end

# constructor that always converts to the correct types
function TGLFNNmodel(fluxmodel::Flux.Chain, name, date, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds)
    return TGLFNNmodel(
        fluxmodel,
        String(name),
        date,
        String.(xnames),
        String.(ynames),
        Float32.(reshape(xm, length(xm))),
        Float32.(reshape(xσ, length(xσ))),
        Float32.(reshape(ym, length(ym))),
        Float32.(reshape(yσ, length(yσ))),
        Float32.(xbounds),
        Float32.(ybounds)
    )
end

# constructor where the date is always filled out
function TGLFNNmodel(fluxmodel::Flux.Chain, name, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds)
    date = Dates.now()
    return TGLFNNmodel(fluxmodel, name, date, xnames, ynames, xm, xσ, ym, yσ, xbounds, ybounds)
end

# TGLFNNensemble
struct TGLFNNensemble <: TGLFmodel
    models::Vector{TGLFNNmodel}
end

function TGLFNNensemble(models::Vector{<:Any})
    return TGLFNNensemble(TGLFNNmodel[model for model in models])
end

function Base.getproperty(ensemble::TGLFNNensemble, field::Symbol)
    if field == :models
        return getfield(ensemble, field)
    elseif field == :fluxmodel
        error("Running TLGF ensemble like a model")
    else
        return getfield(ensemble.models[1], field)
    end
end

#= ============== =#
#  saving/loading  #
#= ============== =#
function mod2dict(model::TGLFNNmodel)
    savedict = Dict()
    for name in fieldnames(TGLFNNmodel)
        value = getproperty(model, name)
        savedict[name] = value
    end
    return savedict
end

function mod2dict(ensemble::TGLFNNensemble)
    savedict = Dict()
    for (km, model) in enumerate(ensemble.models)
        savedict[km] = mod2dict(model)
    end
    return savedict
end

function savemodel(model::TGLFmodel, filename::AbstractString)
    if !endswith(filename, ".bson")
        filename = "$(filename).bson"
    end
    if startswith(filename, "/")
        fullpath = filename
    else
        fullpath = dirname(@__DIR__) * "/models/NN_ensembles/" * filename
    end
    BSON.bson(fullpath, mod2dict(model))
    return fullpath
end

Memoize.@memoize function loadmodelonce(filename::String)
    return loadmodel(filename)
end

function dict2mod(dict::AbstractDict)
    args = []
    for name in fieldnames(TGLFNNmodel)
        push!(args, dict[name])
    end
    return TGLFNNmodel(args...)
end

function dict2ens(dict::Dict)
    return TGLFNNensemble([dict2mod(modict) for modict in values(dict)])
end

function loadmodel(filename::AbstractString)
    if !endswith(filename, ".bson")
        filename = "$(filename).bson"
    end
    if startswith(filename, "/")
        fullpath = filename
    else
        fullpath = dirname(@__DIR__) * "/models/" * filename
        if !isfile(fullpath)
            error("TGLFNN model $filename does not exist. Possible nn models are:\n$(join(readdir(dirname(@__DIR__) * "/models/"),"\n",))")
        end
    end
    savedict = BSON.load(fullpath, @__MODULE__)
    if typeof(first(keys(savedict))) <: Integer
        return dict2ens(savedict)
    else
        return dict2mod(savedict)
    end
end

#= ==================================== =#
#  functions to get the fluxes solution
#= ==================================== =#
function flux_array(fluxmodel::TGLFNNmodel, x::AbstractMatrix; warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> flux_array(fluxmodel, x0; warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

function flux_array(fluxmodel::TGLFNNmodel, x::AbstractVector; warn_nn_train_bounds::Bool=true, natrhonorm::Bool=false)
    if eltype(x) <: Float32
        x32 = copy(x)
    else
        x32 = Float32.(x)
    end
    for ix in findall(map(name -> contains(name, "_log10"), fluxmodel.xnames))
        try
            x32[ix] = Float32(log10.(x[ix]))
        catch e
            error("$(fluxmodel.xnames[ix]) has value $(x32[ix])")
        end
    end
    if warn_nn_train_bounds # training bounds are on the original data but after log10
        for ix in eachindex(x32)
            if any(x32[ix] .< fluxmodel.xbounds[ix, 1])
                @warn("Extrapolation warning on $(fluxmodel.xnames[ix])=$(minimum(x32[ix,:])) is below bound of $(fluxmodel.xbounds[ix,1])")
            elseif any(x32[ix] .> fluxmodel.xbounds[ix, 2])
                @warn("Extrapolation warning on $(fluxmodel.xnames[ix])=$(maximum(x32[ix,:])) is above bound of $(fluxmodel.xbounds[ix,2])")
            end
        end
    end
    xn = (x32 .- fluxmodel.xm) ./ fluxmodel.xσ
    yn = fluxmodel.fluxmodel(xn)
    y = yn .* fluxmodel.yσ .+ fluxmodel.ym
    if natrhonorm
        # Convert name to index
        name_to_indx = Dict{String,Vector{Float32}}()
        for xi in 1:length(fluxmodel.xnames)
            vals = get!(Vector{Float32}, name_to_indx, fluxmodel.xnames[xi])
            push!(vals, xi)
        end
        taus2 = x[convert(Int, name_to_indx["TAUS_2"][1]), :]
        as2 = x[convert(Int, name_to_indx["AS_2"][1]), :]
        rlns1 = x[convert(Int, name_to_indx["RLNS_1"][1]), :]
        rlts1 = x[convert(Int, name_to_indx["RLTS_1"][1]), :]
        rlns2 = x[convert(Int, name_to_indx["RLNS_2"][1]), :]
        rlts2 = x[convert(Int, name_to_indx["RLTS_2"][1]), :]
        ptot = 1 .+ as2 .* taus2
        adpdr = (rlns1 .+ rlts1) .+ as2 .* taus2 .* (rlns2 .+ rlts2)
        for i in 1:size(adpdr)[1]
            if abs(adpdr[i]) < 0.001
                adpdr[i] = 0.001 * sign(adpdr[i])
            end
        end
        aoverLp = adpdr ./ ptot
        nat = (1 ./ taus2) .* (1 ./ aoverLp) .^ 2 .* (1 ./ (2 .* taus2)) .* (0.557^2)
        for k in 1:size(y)[1]
            y[k, :] ./= nat
        end
    end
    # for iy in 1:length(y)
    #     if any(y[iy].<fluxmodel.ybounds[iy,1])
    #         println("Extrapolation warning on $(fluxmodel.ynames[iy])=$(minimum(y[iy,:])) is below bound of $(fluxmodel.ybounds[iy,1])")
    #     elseif any(y[iy].>fluxmodel.ybounds[iy,2])
    #         println("Extrapolation warning on $(fluxmodel.ynames[iy])=$(maximum(y[iy,:])) is above bound of $(fluxmodel.ybounds[iy,2])")
    #     end
    # end
    return eltype(x).(y)
end

function flux_array(fluxensemble::TGLFNNensemble, x::AbstractArray; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    nmodels = length(fluxensemble.models)
    nouts = length(fluxensemble.models[1].ynames)
    nsamples = size(x)[2]

    tmp = zeros(Float32, nmodels, nouts, nsamples)
    Threads.@threads for k in 1:length(fluxensemble.models)
        tmp[k, :, :] = flux_array(fluxensemble.models[k], x; warn_nn_train_bounds)
    end

    mean, std = StatsBase.mean_and_std(tmp, 1; corrected=true)
    if uncertain && nmodels > 1
        return Measurements.measurement.(mean[1, :, :], std[1, :, :])
    else
        return mean[1, :, :]
    end
end

function flux_array(fluxensemble::TGLFNNensemble, x::AbstractVector; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    nmodels = length(fluxensemble.models)
    nouts = length(fluxensemble.models[1].ynames)

    tmp = zeros(Float32, nmodels, nouts)
    Threads.@threads for k in 1:length(fluxensemble.models)
        tmp[k, :] = flux_array(fluxensemble.models[k], x; warn_nn_train_bounds)
    end

    mean, std = StatsBase.mean_and_std(tmp, 1; corrected=true)
    if uncertain
        return Measurements.measurement.(mean[1, :], std[1, :])
    else
        return mean[1, :]
    end
end

function flux_array(fluxmodel::TGLFmodel, args...; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    args = reshape([k for k in args], (length(args), 1))
    return flux_array(fluxmodel, args; uncertain, warn_nn_train_bounds)
end

function flux_solution(fluxmodel::TGLFmodel, args...; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    return IMAS.flux_solution(flux_array(fluxmodel, collect(args); uncertain, warn_nn_train_bounds)...)
end

#= ======================= =#
# functors for TGLFNNmodel
#= ======================= =#
function (fluxmodel::TGLFmodel)(x::AbstractArray; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    return flux_array(fluxmodel, x; uncertain, warn_nn_train_bounds)
end

function (fluxmodel::TGLFmodel)(args...; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    return flux_solution(fluxmodel, args...; uncertain, warn_nn_train_bounds)
end

#= ========== =#
#  run_tglfnn 
#= ========== =#
"""
    run_tglfnn(input_tglf::InputTGLF; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)

Run TGLFNN starting from a InputTGLF, using a specific `model_filename`.

If the model is an ensemble of NNs, then the output can be uncertain (using the Measurements.jl package).

The warn_nn_train_bounds checks against the standard deviation of the inputs to warn if evaluation is likely outside of training bounds.

Returns a `flux_solution` structure
"""
function run_tglfnn(input_tglf::InputTGLF; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)
    tglfmod = TGLFNN.loadmodelonce(model_filename)
    inputs = zeros(length(tglfmod.xnames))
    for (k, item) in enumerate(tglfmod.xnames)
        item = replace(item, "_log10" => "")
        if item == "RLNS_12"
            value = sqrt(input_tglf.RLNS_1^2 + input_tglf.RLNS_2^2)
        else
            value = getfield(input_tglf, Symbol(item))
        end
        # display("input.tglf[$item] -> $value")
        inputs[k] = value
    end
    sol = tglfmod(inputs...; uncertain, warn_nn_train_bounds)
    return sol
end

"""
    run_tglfnn(input_tglfs::Vector{InputTGLF}; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)

Run TGLFNN for multiple InputTGLF, using a specific `model_filename`.

This is more efficient than running TGLFNN on each individual InputTGLFs.

If the model is an ensemble of NNs, then the output can be uncertain (using the Measurements.jl package).

The warn_nn_train_bounds checks against the standard deviation of the inputs to warn if evaluation is likely outside of training bounds.

Returns a vector of `flux_solution` structures
"""
function run_tglfnn(input_tglfs::Vector{InputTGLF}; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)
    tglfmod = TGLFNN.loadmodelonce(model_filename)
    inputs = zeros(length(tglfmod.xnames), length(input_tglfs))
    for (i, input_tglf) in enumerate(input_tglfs)
        for (k, item) in enumerate(tglfmod.xnames)
            item = replace(item, "_log10" => "")
            if item == "RLNS_12"
                value = sqrt(input_tglf.RLNS_1^2 + input_tglf.RLNS_2^2)
            else
                value = getfield(input_tglf, Symbol(item))
            end
            # display("input.tglf[$item] -> $value")
            inputs[k, i] = value
        end
    end
    tmp = flux_array(tglfmod, inputs; uncertain, warn_nn_train_bounds)
    sol = [IMAS.flux_solution(tmp[:, i]...) for i in eachindex(input_tglfs)]
    return sol
end

"""
    run_tglfnn(data::Dict; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)::Dict

Run TGLFNN from a dictionary, using a specific `model_filename`.

If the model is an ensemble of NNs, then the output can be uncertain (using the Measurements.jl package).

The warn_nn_train_bounds checks against the standard deviation of the inputs to warn if evaluation is likely outside of training bounds.

Returns a dictionary with fluxes
"""
function run_tglfnn(data::Dict; model_filename::String, uncertain::Bool=false, warn_nn_train_bounds::Bool)::Dict
    tglfmod = loadmodelonce(model_filename)
    xnames = [replace(name, "_log10" => "") for name in tglfmod.xnames]
    x = collect(transpose(reduce(hcat, [Float64.(data[name]) for name in xnames])))
    y = tglfmod(x; uncertain, warn_nn_train_bounds)
    ynames = [replace(name, "OUT_" => "") for name in tglfmod.ynames]
    return Dict(name => y[k, :] for (k, name) in enumerate(ynames))
end

export run_tglfnn