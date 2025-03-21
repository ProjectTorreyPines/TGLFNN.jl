import Flux
import Flux: NNlib
import Dates
import Memoize
import StatsBase
import Measurements
import BSON
import ONNXNaiveNASflux
import ONNXRunTime as ORT
using ONNXRunTime.CAPI
using ONNXRunTime: testdatapath


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
    xm::Vector{Float64}
    xσ::Vector{Float64}
    ym::Vector{Float64}
    yσ::Vector{Float64}
    xbounds::Array{Float64}
    ybounds::Array{Float64}
    nions::Int
end

function Base.show(io::IO, mime::MIME"text/plain", model::TGLFNNmodel)
    println(io, "TGLFNNmodel")
    println(io, "name: $(length(model.name))")
    println(io, "date: $(model.date)")
    println(io, "nions: $(model.nions)")
    println(io, "xnames ($(length(model.xnames))): $(model.xnames)")
    return println(io, "ynames ($(length(model.ynames))): $(model.ynames)")
end

# TGLFNNensemble
struct TGLFNNensemble <: TGLFmodel
    models::Vector{TGLFNNmodel}
end

function Base.show(io::IO, mime::MIME"text/plain", ensemble::TGLFNNensemble)
    println(io, "TGLFNNensemble")
    println(io, "n models: $(length(ensemble.models))")
    return show(io, mime, ensemble.models[1])
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

function dict2mod(savedict::AbstractDict)
    args = []
    for name in fieldnames(TGLFNNmodel)
        if name == :fluxmodel
            savedict[name] = Flux.fmap(Flux.f64, savedict[name])
            push!(args, savedict[name])
        elseif name == :nions
            nions = maximum(map(m -> parse(Int, m[1]), filter(!isnothing, match.(r"_([0-9]+$)", savedict[:xnames])))) - 1
            push!(args, nions)
        else
            push!(args, savedict[name])
        end
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
            error("TGLFNN model $filename does not exist. Possible nn models are:\n    $(join(available_models(),"\n    ",))")
        end
    end
    savedict = BSON.load(fullpath, @__MODULE__)
    if typeof(first(keys(savedict))) <: Integer
        return dict2ens(savedict)
    else
        return dict2mod(savedict)
    end
end

function available_models()
    models_dir = joinpath(dirname(@__DIR__), "models")
    return [replace(model, r"\.(bson|onnx)$" => "") for model in readdir(models_dir) if endswith(model, ".bson") || endswith(model, ".onnx")]
end

#= ==================================== =#
#  functions to get the fluxes solution
#= ==================================== =#
function flux_array(fluxmodel::TGLFNNmodel, x::AbstractMatrix; warn_nn_train_bounds::Bool=true)
    return hcat(collect(map(x0 -> flux_array(fluxmodel, x0; warn_nn_train_bounds), eachslice(x; dims=2)))...)
end

function flux_array(fluxmodel::TGLFNNmodel, x::AbstractVector; warn_nn_train_bounds::Bool=true)
    xx = [contains(name, "_log10") ? log10.(x[ix]) : x[ix] for (ix, name) in enumerate(fluxmodel.xnames)]
    if warn_nn_train_bounds # training bounds are on the original data but after log10
        for ix in eachindex(xx)
            if any(xx[ix] .< fluxmodel.xbounds[ix, 1])
                @warn("Extrapolation warning on $(fluxmodel.xnames[ix])=$(minimum(xx[ix,:])) is below bound of $(fluxmodel.xbounds[ix,1])")
            elseif any(xx[ix] .> fluxmodel.xbounds[ix, 2])
                @warn("Extrapolation warning on $(fluxmodel.xnames[ix])=$(maximum(xx[ix,:])) is above bound of $(fluxmodel.xbounds[ix,2])")
            end
        end
    end
    xn = (xx .- fluxmodel.xm) ./ fluxmodel.xσ
    yn = fluxmodel.fluxmodel(xn)
    yy = yn .* fluxmodel.yσ .+ fluxmodel.ym
    return yy
end

function flux_array(fluxensemble::TGLFNNensemble, x::AbstractArray; uncertain::Bool=false, warn_nn_train_bounds::Bool=true)
    nmodels = length(fluxensemble.models)
    nouts = length(fluxensemble.models[1].ynames)
    nsamples = size(x)[2]

    tmp = zeros(nmodels, nouts, nsamples)
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

    tmp = zeros(nmodels, nouts)
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
    return flux_solution(flux_array(fluxmodel, collect(args); uncertain, warn_nn_train_bounds)...)
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
            inputs[k, i] = value
        end
    end
    tmp = flux_array(tglfmod, inputs; uncertain, warn_nn_train_bounds)
    sol = [flux_solution(tmp[:, i]...) for i in eachindex(input_tglfs)]
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

function load_onnx_model(onnx_path::String)
    if !contains(onnx_path, "/models/")
        onnx_path = joinpath(dirname(@__DIR__), "models", onnx_path)
        if !endswith(onnx_path, ".onnx")
            onnx_path *= ".onnx"
        end
        if !isfile(onnx_path)
            error("TGLFNN model does not exist in $onnx_path")
        end
    end
    return ORT.load_inference(ORT.testdatapath(onnx_path))
end

function build_input_value(input_tglf::InputTGLF, name::String)
    key = replace(name, "_log10" => "")
    value = key == "RLNS_12" ? sqrt(input_tglf.RLNS_1^2 + input_tglf.RLNS_2^2) : getfield(input_tglf, Symbol(key))
    return occursin("_log10", name) ? log10(value) : value
end

function build_inputs(input_tglfs::Vector{InputTGLF}, xnames::Vector{String})
    return hcat([[build_input_value(t, x) for x in xnames] for t in input_tglfs]...)
end

# Reorder output rows to match a new order, e.g. [1, 4, 2, 3]
function reorder_output(out::AbstractMatrix, order::Vector{Int})
    return reduce(hcat, [out[i, :] for i in order])'
end

function run_tglfnn_onnx(input_tglfs::Vector{InputTGLF}, onnx_path::String, xnames::Vector{String}, ynames::Vector{String})
    model = load_onnx_model(onnx_path)
    inputs = build_inputs(input_tglfs, xnames)
    tmp = model(Dict("input" => Float32.(inputs')))["output"]'
    tmp_new = reorder_output(tmp, [1, 4, 2, 3])
    sol = [flux_solution(tmp_new[:, i]...) for i in 1:size(tmp_new, 2)]
    return sol
end

function run_tglfnn_onnx(data::Dict, onnx_path::String, xnames::Vector{String}, ynames::Vector{String})::Dict
    model = load_onnx_model(onnx_path)
    xnames_clean = [replace(name, "_log10" => "") for name in xnames]
    x = reduce(hcat, [Float64.(data[name]) for name in xnames_clean])
    x = Float32.(x')
    y = model(Dict("input" => x))["output"]'
    ynames_clean = [replace(name, "OUT_" => "") for name in ynames]
    return Dict(name => y[k, :] for (k, name) in enumerate(ynames_clean))
end

function run_tglfnn_onnx(input_tglf::InputTGLF, onnx_path::String, xnames::Vector{String}, ynames::Vector{String})
    model = load_onnx_model(onnx_path)
    values = [build_input_value(input_tglf, x) for x in xnames]
    sol = model(Dict("input" => Float32.([values]')))["output"]'
    sol_new = [sol[1], sol[4], sol[2], sol[3]]
    return sol_new
end

"""
    flux_solution(xx::Vararg{T}) where {T<:Real}

Constructor used to handle PARTICLE_FLUX_i entered as a set of scalars instead of an array

    flux_solution(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

results in

    Qe = 1.0
    Qi = 2.0
    Γe = 3.0
    Γi = [4.0, 5.0]
    Πi = 6.0

NOTE: for backward compatibility with old TGLF-NN models, if number of arguments is 4 then

    flux_solution(1.0, 2.0, 3.0, 4.0)

results in

    Qe = 3.0
    Qi = 4.0
    Γe = 1.0
    Γi = []
    Πi = 2.0
"""
function flux_solution(xx::Vararg{T}) where {T<:Real}
    n_fields = length(xx)
    if n_fields == 4
        ENERGY_FLUX_e = 3
        ENERGY_FLUX_i = 4
        PARTICLE_FLUX_e = 1
        STRESS_TOR_i = 2
        sol = GACODE.FluxSolution(xx[ENERGY_FLUX_e], xx[ENERGY_FLUX_i], xx[PARTICLE_FLUX_e], T[], xx[STRESS_TOR_i])
    else
        ENERGY_FLUX_e = n_fields - 1
        ENERGY_FLUX_i = n_fields
        PARTICLE_FLUX_e = 1
        PARTICLE_FLUX_i = 2:n_fields-3
        STRESS_TOR_i = n_fields - 2
        sol = GACODE.FluxSolution(xx[ENERGY_FLUX_e], xx[ENERGY_FLUX_i], xx[PARTICLE_FLUX_e], T[xx[i] for i in PARTICLE_FLUX_i], xx[STRESS_TOR_i])
    end
    return sol
end

export run_tglfnn, run_tglfnn_onnx