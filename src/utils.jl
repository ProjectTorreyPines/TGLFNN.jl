"""
    save(input::Union{InputTGLF, InputCGYRO}, filename::AbstractString)

Write input_tglf/input_cgyro to file in input.tglf/input.cgyro/input.qlgyro format to be read by TGLF/CGYRO
"""
function save(input::Union{InputTGLF,InputCGYRO,InputQLGYRO}, filename::AbstractString)
    open(filename, "w") do io
        for key in fieldnames(typeof(input))
            if startswith(String(key), "_")
                continue
            end
            try
                value = getfield(input, key)
                if ismissing(value)
                    continue
                elseif isa(value, Int)
                    println(io, "$(key)=$(convert(Int, value))")
                elseif isa(value, String)
                    println(io, "$(key)='$value'")
                elseif isa(value, Bool)
                    println(io, "$(key)=.$value.")
                else
                    println(io, "$(key)=$(convert(Float64, value))")
                end
            catch e
                println("Error writing $key to input file")
                rethrow(e)
            end
        end
    end
end

"""
    compare_two_input_tglfs(itp_1::InputTGLF, itp_2::InputTGLF)

Compares two input_tglfs, prints the difference and stores the difference in a new InputTGLF
"""
function compare_two_input_tglfs(itp_1::InputTGLF, itp_2::InputTGLF)
    itp_diff = InputTGLF()
    for field in fieldnames(typeof(itp_diff))
        if typeof(getproperty(itp_1, field)) <: String
            setproperty!(itp_diff, field, getproperty(itp_1, field) * "  " * getproperty(itp_2, field))
        else
            setproperty!(itp_diff, field, getproperty(itp_1, field) - getproperty(itp_2, field))
        end
    end

    for key in fieldnames(typeof(itp_diff))
        itp_1_value = getproperty(itp_1, key)
        itp_diff_value = getproperty(itp_diff, key)
        if typeof(itp_diff_value) <: String || typeof(itp_diff_value) <: Missing
            continue
        end
        println("Difference for $key = $(round(itp_diff_value/itp_1_value*100,digits=2)) % itp_2 = $(itp_diff_value+itp_1_value), itp_1 = $itp_1_value ")
    end

    return itp_diff
end

function diff(A::InputTGLF, B::InputTGLF)
    differences = Symbol[]
    for field in fieldnames(typeof(A))
        if getfield(A, field) === getfield(B, field)
            # pass
        else
            push!(differences, field)
        end
    end
    return differences
end

function scan(input_tglf::InputTGLF; kw...)
    # Convert keyword arguments to a dictionary for easier handling
    kw_dict = Dict(kw)

    # Base case: if no keywords are left, return the current input_tglf
    if isempty(kw_dict)
        return [deepcopy(input_tglf)]
    end

    # Extract one keyword and its values
    key, values = pop!(kw_dict)
    intglfs = InputTGLF[]

    # Iterate over the values for the current keyword
    for v in values
        tmp = deepcopy(input_tglf)
        setproperty!(tmp, key, v)

        # Recursively call scan for the rest of the keywords
        intglfs = vcat(intglfs, scan(tmp; kw_dict...))
    end

    return intglfs
end

export compare_two_input_tglfs

"""
    parse_out_tglf_gbflux(lines::String; outnames=["Gam/Gam_GB", "Q/Q_GB", "Pi/Pi_GB", "S/S_GB"])

parse out.tglf.gbflux file into a dictionary with possibility of using custom names for outputs
"""
function parse_out_tglf_gbflux(lines::String; outnames::NTuple{4,String}=("Gam/Gam_GB", "Q/Q_GB", "Pi/Pi_GB", "S/S_GB"))
    vals = map(x -> parse(Float64, x), split.(lines))
    ns = Int(length(vals) / length(outnames))
    out = Dict()
    let k = 1
        for t in outnames
            out[t*"_elec"] = vals[k]
            k += 1
            out[t*"_all_ions"] = Float64[]
            for s in 1:ns-1
                out[t*"_ion$s"] = vals[k]
                push!(out[t*"_all_ions"], vals[k])
                k += 1
            end
            out[t*"_ions"] = sum(out[t*"_all_ions"])
        end
    end
    return out
end
