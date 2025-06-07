mutable struct InputTGLFs
    tglfs::Vector{InputTGLF}
end

function Base.setproperty!(inputTGLFs::InputTGLFs, field::Symbol, value::AbstractVector{<:Any})
    tglfs = getfield(inputTGLFs, :tglfs)
    @assert length(value) == length(tglfs)
    for (k, inputTGLF) in enumerate(tglfs)
        setproperty!(inputTGLF, field, value[k])
    end
    return value
end

function Base.setproperty!(inputTGLFs::InputTGLFs, field::Symbol, value::Any)
    for (k, inputTGLF) in enumerate(getfield(inputTGLFs, :tglfs))
        setproperty!(inputTGLF, field, value)
    end
    return value
end

function Base.getproperty(inputTGLFs::InputTGLFs, field::Symbol)
    if field === :tglfs
        return getfield(inputTGLFs, :tglfs)
    else
        tglfs = getfield(inputTGLFs, :tglfs)
        FT = fieldtype(typeof(tglfs[1]), field)
        data = Vector{FT}(undef, length(tglfs))
        for (k, inputTGLF) in enumerate(tglfs)
            data[k] = getproperty(inputTGLF, field)
        end
        return data
    end
end

function Base.getindex(inputTGLFs::InputTGLFs, index::Int)
    return getfield(inputTGLFs, :tglfs)[index]
end
