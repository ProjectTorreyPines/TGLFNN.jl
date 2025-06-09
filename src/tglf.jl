function Base.show(io::IO, ::MIME"text/plain", input_tglf::InputTGLF)
    for fname in sort!(collect(fieldnames(typeof(input_tglf))))
        value = getfield(input_tglf, fname)
        if value !== missing && (!isdigit(string(fname)[end]) || (isdigit(string(fname)[end]) && parse(Int, split(string(fname), "_")[end]) <= input_tglf.NS))
            println(io, "$fname = $(value)")
        end
    end
end

"""
    InputTGLF(dd::IMAS.dd, rho::AbstractVector{Float64}, sat::Symbol=:sat0, electromagnetic::Bool=false, lump_ions::Bool=true)

Evaluate TGLF input parameters at given radii
"""
function InputTGLF(dd::IMAS.dd, rho::AbstractVector{Float64}, sat::Symbol=:sat0, electromagnetic::Bool=false, lump_ions::Bool=true)
    eqt = dd.equilibrium.time_slice[]
    cp1d = dd.core_profiles.profiles_1d[]
    gridpoint_cp = [argmin_abs(cp1d.grid.rho_tor_norm, ρ) for ρ in rho]
    return InputTGLF(eqt, cp1d, gridpoint_cp, sat, electromagnetic, lump_ions)
end

"""
    InputTGLF(dd::IMAS.dd, gridpoint_cp::AbstractVector{Int}, sat::Symbol=:sat0, electromagnetic::Bool=false, lump_ions::Bool=true)

Evaluate TGLF input parameters at given core profiles grid indexes
"""
function InputTGLF(dd::IMAS.dd, gridpoint_cp::AbstractVector{Int}, sat::Symbol=:sat0, electromagnetic::Bool=false, lump_ions::Bool=true)
    eqt = dd.equilibrium.time_slice[]
    cp1d = dd.core_profiles.profiles_1d[]
    return InputTGLF(eqt, cp1d, gridpoint_cp, sat, electromagnetic, lump_ions)
end

function InputTGLF(
    eqt::IMAS.equilibrium__time_slice,
    cp1d::IMAS.core_profiles__profiles_1d,
    gridpoint_cp::AbstractVector{Int},
    sat::Symbol,
    electromagnetic::Bool,
    lump_ions::Bool)

    e = IMAS.cgs.e # statcoul
    k = IMAS.cgs.k # erg/eV
    mp = IMAS.cgs.mp # g
    me = IMAS.cgs.me # g
    md = IMAS.cgs.md # g
    m_to_cm = IMAS.cgs.m_to_cm
    m³_to_cm³ = IMAS.cgs.m³_to_cm³
    T_to_Gauss = IMAS.cgs.T_to_Gauss

    eqt1d = eqt.profiles_1d
    rho_cp = cp1d.grid.rho_tor_norm
    rho_eq = eqt1d.rho_tor_norm

    if lump_ions
        ions = IMAS.lump_ions_as_bulk_and_impurity(cp1d)
    else
        ions = cp1d.ion
    end

    Rmaj = IMAS.interp1d(rho_eq, m_to_cm * 0.5 * (eqt1d.r_outboard .+ eqt1d.r_inboard)).(rho_cp)

    rmin = GACODE.r_min_core_profiles(eqt1d, rho_cp)

    q_profile = IMAS.interp1d(rho_eq, eqt1d.q).(rho_cp)

    if !ismissing(eqt1d, :elongation)
        kappa = IMAS.interp1d(rho_eq, eqt1d.elongation).(rho_cp)
    else
        kappa = zero(rho_cp)
    end

    if !ismissing(eqt1d, :triangularity_lower) && !ismissing(eqt1d, :triangularity_upper)
        delta = IMAS.interp1d(rho_eq, 0.5 * (eqt1d.triangularity_lower + eqt1d.triangularity_upper)).(rho_cp)
    else
        delta = zero(rho_cp)
    end

    if !ismissing(eqt1d, :squareness_lower_inner) && !ismissing(eqt1d, :squareness_lower_outer) && !ismissing(eqt1d, :squareness_upper_inner) &&
       !ismissing(eqt1d, :squareness_upper_outer)
        tmp = 0.25 .* (eqt1d.squareness_lower_inner .+ eqt1d.squareness_lower_outer .+ eqt1d.squareness_upper_inner .+ eqt1d.squareness_upper_outer)
        zeta = IMAS.interp1d(rho_eq, tmp).(rho_cp)
    else
        zeta = zero(rho_cp)
    end

    a = rmin[end]
    @views q = q_profile[gridpoint_cp]

    Te_full = cp1d.electrons.temperature
    dlntedr_full = .-IMAS.calc_z(rmin, Te_full, :backward)
    @views Te = Te_full[gridpoint_cp]
    @views dlntedr = dlntedr_full[gridpoint_cp]

    ne_full = cp1d.electrons.density_thermal ./ m³_to_cm³
    dlnnedr_full = .-IMAS.calc_z(rmin, ne_full, :backward)
    @views ne = ne_full[gridpoint_cp]
    @views dlnnedr = dlnnedr_full[gridpoint_cp]

    Bt = eqt.global_quantities.vacuum_toroidal_field.b0
    buitp = IMAS.interp1d(rho_eq, GACODE.bunit(eqt1d))
    bunit = @. @views buitp(rho_cp[gridpoint_cp]) * T_to_Gauss
    input_tglfs = InputTGLFs([InputTGLF() for k in eachindex(gridpoint_cp)])

    signb = sign(Bt)
    input_tglfs.SIGN_BT = signb
    getview(input_tglfs, :SIGN_IT) .= @. signb * sign(q)

    input_tglfs.NS = length(ions) + 1 # add 1 to include electrons

    # electrons first for TGLF
    input_tglfs.MASS_1 = me / md
    input_tglfs.TAUS_1 = 1.0
    input_tglfs.AS_1 = 1.0
    input_tglfs.ZS_1 = -1.0
    getview(input_tglfs, :RLNS_1) .= @. a * dlnnedr
    getview(input_tglfs, :RLTS_1) .= @. a * dlntedr

    c_s = GACODE.c_s.(Te)
    w0 = @. -cp1d.rotation_frequency_tor_sonic
    w0p = IMAS.gradient(rmin, w0)
    gamma_p = @. @views -Rmaj[gridpoint_cp] * w0p[gridpoint_cp]
    gamma_e = @. @views -rmin[gridpoint_cp] / q * w0p[gridpoint_cp]
    mach = @. @views Rmaj[gridpoint_cp] * w0[gridpoint_cp] / c_s
    getview(input_tglfs, :VPAR_1) .= @. -input_tglfs.SIGN_IT * mach
    getview(input_tglfs, :VPAR_SHEAR_1) .= @. -input_tglfs.SIGN_IT * (a / c_s) * gamma_p
    getview(input_tglfs, :VEXB_SHEAR) .= @. -gamma_e * (a / c_s)

    for iion in eachindex(ions)
        species = iion + 1
        Ti_full = ions[iion].temperature
        dlntidr_full = .-IMAS.calc_z(rmin, Ti_full, :backward)
        @views Ti = Ti_full[gridpoint_cp]
        @views dlntidr = dlntidr_full[gridpoint_cp]

        Zi = IMAS.avgZ(ions[iion].element[1].z_n, Ti)
        setproperty!(input_tglfs, Symbol("ZS_$species"), Zi)
        setproperty!(input_tglfs, Symbol("MASS_$species"), ions[iion].element[1].a .* mp ./ md)

        ni_full = ions[iion].density_thermal ./ m³_to_cm³
        dlnnidr_full = .-IMAS.calc_z(rmin, ni_full, :backward)
        @views ni = ni_full[gridpoint_cp]
        @views dlnnidr = dlnnidr_full[gridpoint_cp]

        setproperty!(input_tglfs, Symbol("TAUS_$species"), Ti ./ Te)
        setproperty!(input_tglfs, Symbol("AS_$species"), ni ./ ne)
        setproperty!(input_tglfs, Symbol("VPAR_$species"), input_tglfs.VPAR_1)
        setproperty!(input_tglfs, Symbol("VPAR_SHEAR_$species"), input_tglfs.VPAR_SHEAR_1)
        setproperty!(input_tglfs, Symbol("RLNS_$species"), a .* dlnnidr)
        setproperty!(input_tglfs, Symbol("RLTS_$species"), a .* dlntidr)
    end

    getview(input_tglfs, :BETAE) .= @. 8π * ne * k * Te / bunit^2
    getview(input_tglfs, :XNUE) .= @. a / c_s * sqrt(ions[1].element[1].a) * e^4 * π * ne * (24.0 - log(sqrt(ne) / Te)) / (sqrt(me) * (k * Te)^1.5)
    getview(input_tglfs, :ZEFF) .= @views cp1d.zeff[gridpoint_cp]
    rho_s = @views GACODE.rho_s(cp1d, eqt)[gridpoint_cp]
    getview(input_tglfs, :DEBYE) .= @. 7.43e2 * sqrt(Te / ne) / rho_s
    getview(input_tglfs, :RMIN_LOC) .= @. @views rmin[gridpoint_cp] / a
    getview(input_tglfs, :RMAJ_LOC) .= @. @views Rmaj[gridpoint_cp] / a
    input_tglfs.ZMAJ_LOC = 0
    input_tglfs.DRMINDX_LOC = 1.0

    drmaj = IMAS.gradient(rmin, Rmaj)

    input_tglfs.DRMAJDX_LOC = @views drmaj[gridpoint_cp]
    input_tglfs.DZMAJDX_LOC = 0.0

    getview(input_tglfs, :Q_LOC) .= @. abs(q)

    input_tglfs.KAPPA_LOC = @views kappa[gridpoint_cp]

    skappa = rmin .* IMAS.gradient(rmin, kappa) ./ kappa
    sdelta = rmin .* IMAS.gradient(rmin, delta)
    szeta = rmin .* IMAS.gradient(rmin, zeta)

    input_tglfs.S_KAPPA_LOC = @views skappa[gridpoint_cp]
    input_tglfs.DELTA_LOC = @views delta[gridpoint_cp]
    input_tglfs.S_DELTA_LOC = @views sdelta[gridpoint_cp]
    input_tglfs.ZETA_LOC = @views zeta[gridpoint_cp]
    input_tglfs.S_ZETA_LOC = @views szeta[gridpoint_cp]

    press = cp1d.pressure_thermal
    Pa_to_dyn = 10.0

    dpdr = @views IMAS.gradient(rmin, press)[gridpoint_cp] .* Pa_to_dyn
    getview(input_tglfs, :P_PRIME_LOC) .= @. @views abs(q) / (rmin[gridpoint_cp] / a)^2 * rmin[gridpoint_cp] / bunit^2 * dpdr

    dqdr = @views IMAS.gradient(rmin, q_profile)[gridpoint_cp]
    getview(input_tglfs, :Q_PRIME_LOC) .= @. @views q * a^2 / rmin[gridpoint_cp] * dqdr

    # saturation rules
    input_tglfs.ALPHA_ZF = 1.0 # 1 = default, -1 = low ky cutoff kypeak search
    input_tglfs.USE_MHD_RULE = false
    input_tglfs.NMODES = input_tglfs.NS .+ 2 # capture main branches: ES each species + BPER + VPAR_SHEAR
    input_tglfs.NKY = 12 # 12 is default, 16 for smoother spectrum
    input_tglfs.ALPHA_QUENCH = 0 # 0 = spectral shift, 1 = quench
    input_tglfs.SAT_RULE = parse(Int, split(string(sat), "sat")[end])
    if sat == :sat2 || sat == :sat3
        input_tglfs.UNITS = "CGYRO"
        input_tglfs.KYGRID_MODEL = 4
        input_tglfs.NBASIS_MIN = 2
        input_tglfs.NBASIS_MAX = 6
        input_tglfs.USE_AVE_ION_GRID = true
        input_tglfs.XNU_MODEL = 3
        input_tglfs.WDIA_TRAPPED = 1.0
    else
        input_tglfs.UNITS = "GYRO"
        if sat == :sat1
        elseif sat == :sat1geo
            input_tglfs.UNITS = "CGYRO"
        elseif sat == :sat0quench
            input_tglfs.ALPHA_QUENCH = 1
        end
        input_tglfs.KYGRID_MODEL = 1
        input_tglfs.NBASIS_MIN = 2
        input_tglfs.NBASIS_MAX = 4
        input_tglfs.USE_AVE_ION_GRID = false # default is false
        input_tglfs.XNU_MODEL = 2
        input_tglfs.WDIA_TRAPPED = 0.0
    end

    # electrostatic/electromagnetic
    if electromagnetic
        input_tglfs.USE_BPER = true
        input_tglfs.USE_BPAR = true
    else
        input_tglfs.USE_BPER = false
        input_tglfs.USE_BPAR = false
    end

    input_tglfs.ALPHA_MACH = 0.0
    return input_tglfs
end

"""
    load(input_tglf::InputTGLF, filename::AbstractString)

Reads filename (`input.tglf` or `input.tglf.gen` format) and populates input_tglf
"""
function load(input_tglf::InputTGLF, filename::String)
    lines = open(filename, "r") do file
        return filter(x -> length(x) > 0, map(strip, split(read(file, String), "\n")))
    end

    ip_dict = Dict()
    if all(line -> contains(line, "="), lines)
        for line in lines
            field, value = map(strip, split(line, "="))
            ip_dict[Symbol(field)] = value
        end
    elseif !all(line -> contains(line, "="), lines)
        for line in lines
            value, field = map(strip, split(line, "  "))
            ip_dict[Symbol(field)] = value
        end
    else
        error("invalid input.tglf or input.tglf.gen file")
    end

    field_types = fieldtypes(InputTGLF)
    for (idx, field) in enumerate(fieldnames(typeof(input_tglf)))
        if typeof(field_types[idx]) <: Union
            type_of_item = field_types[idx].b
        else
            type_of_item = field_types[idx]
        end
        if field ∉ keys(ip_dict)
            continue
        end
        if ip_dict[field] == "T" || ip_dict[field] == ".true."
            setproperty!(input_tglf, field, true)
        elseif type_of_item == "F" || ip_dict[field] == ".false."
            setproperty!(input_tglf, field, false)
        elseif type_of_item <: Float64
            setproperty!(input_tglf, field, parse(Float64, ip_dict[field]))
        elseif type_of_item <: Int64
            setproperty!(input_tglf, field, Int(parse(Float64, ip_dict[field])))
        elseif type_of_item <: String
            setproperty!(input_tglf, field, ip_dict[field])
        else
            error("parameter $field of type ($type_of_item) not recognized: $(ip_dict[field])")
        end
    end
    return input_tglf
end

"""
    run_tglf(input_tglf::InputTGLF)

Run TGLF starting from a InputTGLF.

Returns a `FluxSolution` structure
"""
function run_tglf(input_tglf::InputTGLF)
    folder = mktempdir()

    save(input_tglf, joinpath(folder, "input.tglf"))

    open(joinpath(folder, "command.sh"), "w") do io
        return write(
            io,
            """
         if command -v timeout &> /dev/null; then
         	(time (timeout 120 tglf -n 1 -e .)) &> command.log
         else
         	(time (tglf -n 1 -e .)) &> command.log
         fi
         """)
    end

    fluxes = try
        run(Cmd(`bash command.sh`; dir=folder))

        tmp = open(joinpath(folder, "out.tglf.gbflux"), "r") do io
            return read(io, String)
        end

        parse_out_tglf_gbflux(tmp)

    catch e
        # show last 100 lines of  chease.output
        txt = open(joinpath(folder, "command.log"), "r") do io
            return split(read(io, String), "\n")
        end
        @error "ERROR running TGLF\n...\n" * join(txt[max(1, length(txt) - 100):end], "\n")
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
    run_tglf(input_tglf::InputTGLF)

Run TGLF starting from a vector of InputTGLFs.

NOTE: Each run is done asyncronously (ie. in separate parallel processes)

Returns a `FluxSolution` structure
"""
function run_tglf(input_tglfs::Vector{InputTGLF})
    return collect(asyncmap(input_tglf -> run_tglf(input_tglf), input_tglfs))
end

export run_tglf
