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


"""
    FieldView(parent, field) <: AbstractVector

A one-dimensional view of a single field `field` across all
`InputTGLF`s stored in `parent::InputTGLFs`.
No data are copied: every access delegates to `getproperty` /
`setproperty!` on the corresponding struct.
"""
struct FieldView{T} <: AbstractVector{T}
    parent :: InputTGLFs
    field  :: Symbol
end

# -- AbstractVector interface -------------------------------------------
Base.IndexStyle(::Type{<:FieldView}) = IndexLinear()
Base.size(v::FieldView)    = (length(v.parent.tglfs),)
Base.length(v::FieldView)  = length(v.parent.tglfs)

@inline Base.getindex(v::FieldView, i::Int) =
    getproperty(v.parent.tglfs[i], v.field)

@inline function Base.setindex!(v::FieldView, x, i::Int)
    setproperty!(v.parent.tglfs[i], v.field, x)
end

# Tell broadcast we are already a container
Base.broadcastable(v::FieldView) = v

"""
    getview(A::InputTGLFs, field::Symbol) -> FieldView

Return a proxy vector that points to `field` inside each element of `A.tglfs`.
"""
function getview(A::InputTGLFs, field::Symbol)
    FT = fieldtype(InputTGLF, field)
    return FieldView{FT}(A, field)
end