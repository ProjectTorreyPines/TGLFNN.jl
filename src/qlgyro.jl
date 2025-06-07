Base.@kwdef mutable struct InputQLGYRO
    N_PARALLEL::Union{Int,Missing} = missing
    N_RUNS::Union{Int,Missing} = missing
    GAMMA_E::Union{Float64,Missing} = missing
    CODE::Union{Int,Missing} = missing
    NKY::Union{Int,Missing} = missing
    KYGRID_MODEL::Union{Int,Missing} = missing
    KY::Union{Float64,Missing} = missing
    SAT_RULE::Union{Int,Missing} = missing
end

"""
    input_qlgyro(input_qlgyro::InputQLGYRO, input_cgyro::InputCGYRO)

Run QLGYRO starting from a InputQLGYRO and InputCGYRO

Returns a `FluxSolution` structure
"""
function run_qlgyro(input_qlgyro::InputQLGYRO, input_cgyro::InputCGYRO)
    folder = mktempdir()

    save(input_cgyro, joinpath(folder, "input.cgyro"))
    save(input_qlgyro, joinpath(folder, "input.qlgyro"))

    n_parallel = input_qlgyro.N_PARALLEL
    open(joinpath(folder, "command.sh"), "w") do io
        return write(
            io,
            """
         	(time (qlgyro -n $n_parallel -e .)) &> command.log
         """)
    end

    fluxes = try
        run(Cmd(`bash command.sh`; dir=folder))

        tmp = open(joinpath(folder, "out.qlgyro.gbflux"), "r") do io
            return read(io, String)
        end
        fluxes = parse_out_tglf_gbflux(tmp)

    catch e
        # show last 100 lines of  chease.output
        txt = open(joinpath(folder, "command.log"), "r") do io
            return split(read(io, String), "\n")
        end
        @error "ERROR running QLGYRO\n...\n" * join(txt[max(1, length(txt) - 100):end], "\n")
        rethrow(e)
    end

    sol = GACODE.FluxSolution(
        fluxes["Q/Q_GB_elec"],
        fluxes["Q/Q_GB_ions"],
        fluxes["Gam/Gam_GB_elec"],
        fluxes["Gam/Gam_GB_all_ions"],
        fluxes["Pi/Pi_GB_ions"])

    rm(folder; recursive=true)

    return sol
end

"""
    run_qlgyro(input_qlgyros::Vector{InputQLGYRO}, input_cgyros::Vector{InputCGYRO})

Run QLGYRO starting from a vectors of InputQLGYRO and InputCGYRO

NOTE: Each run is done sequentially, one after the other

Returns a vector of `FluxSolution` structures
"""
function run_qlgyro(input_qlgyros::Vector{InputQLGYRO}, input_cgyros::Vector{InputCGYRO})
    @assert length(input_qlgyros) == length(input_cgyros)
    return collect(map((input_qlgyro, input_cgyro) -> TGLFNN.run_qlgyro(input_qlgyro, input_cgyro), input_qlgyros, input_cgyros))
end

export run_qlgyro
