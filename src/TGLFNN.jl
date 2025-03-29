module TGLFNN

using IMAS
import GACODE

include("tglf.jl")

include("tglf_nn.jl")

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end # module
