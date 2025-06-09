module TGLFNN

using IMAS
using IMASutils: argmin_abs
import TJLF
import TJLF: InputTGLF, InputTJLF
import GACODE

include("tglf.jl")

include("input_tglfs.jl")

include("tjlf.jl")

include("tglf_nn.jl")

include("cgyro.jl")

include("qlgyro.jl")

include("utils.jl")

export InputTGLF, InputTJLF

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end # module
