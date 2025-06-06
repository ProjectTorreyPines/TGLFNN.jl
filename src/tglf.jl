Base.@kwdef mutable struct InputTGLF
    SIGN_BT::Union{Int,Missing} = missing
    SIGN_IT::Union{Int,Missing} = missing
    NS::Union{Int,Missing} = missing
    ZMAJ_LOC::Union{Float64,Missing} = missing
    DRMINDX_LOC::Union{Float64,Missing} = missing
    DZMAJDX_LOC::Union{Float64,Missing} = missing
    S_DELTA_LOC::Union{Float64,Missing} = missing
    ZETA_LOC::Union{Float64,Missing} = missing
    S_ZETA_LOC::Union{Float64,Missing} = missing

    MASS_1::Union{Float64,Missing} = missing
    ZS_1::Union{Float64,Missing} = missing
    AS_1::Union{Float64,Missing} = missing
    TAUS_1::Union{Float64,Missing} = missing

    MASS_2::Union{Float64,Missing} = missing
    ZS_2::Union{Float64,Missing} = missing
    VPAR_2::Union{Float64,Missing} = missing
    VPAR_SHEAR_2::Union{Float64,Missing} = missing

    MASS_3::Union{Float64,Missing} = missing
    ZS_3::Union{Float64,Missing} = missing
    RLTS_3::Union{Float64,Missing} = missing
    TAUS_3::Union{Float64,Missing} = missing
    VPAR_3::Union{Float64,Missing} = missing
    VPAR_SHEAR_3::Union{Float64,Missing} = missing

    # TGLF-NN uses 3 species
    # This is why parameters for species 1:3 are sorted differently than 4:10
    MASS_4::Union{Float64,Missing} = missing
    AS_4::Union{Float64,Missing} = missing
    ZS_4::Union{Float64,Missing} = missing
    RLNS_4::Union{Float64,Missing} = missing
    RLTS_4::Union{Float64,Missing} = missing
    TAUS_4::Union{Float64,Missing} = missing
    VPAR_4::Union{Float64,Missing} = missing
    VPAR_SHEAR_4::Union{Float64,Missing} = missing

    MASS_5::Union{Float64,Missing} = missing
    AS_5::Union{Float64,Missing} = missing
    ZS_5::Union{Float64,Missing} = missing
    RLNS_5::Union{Float64,Missing} = missing
    RLTS_5::Union{Float64,Missing} = missing
    TAUS_5::Union{Float64,Missing} = missing
    VPAR_5::Union{Float64,Missing} = missing
    VPAR_SHEAR_5::Union{Float64,Missing} = missing

    MASS_6::Union{Float64,Missing} = missing
    AS_6::Union{Float64,Missing} = missing
    ZS_6::Union{Float64,Missing} = missing
    RLNS_6::Union{Float64,Missing} = missing
    RLTS_6::Union{Float64,Missing} = missing
    TAUS_6::Union{Float64,Missing} = missing
    VPAR_6::Union{Float64,Missing} = missing
    VPAR_SHEAR_6::Union{Float64,Missing} = missing

    MASS_7::Union{Float64,Missing} = missing
    AS_7::Union{Float64,Missing} = missing
    ZS_7::Union{Float64,Missing} = missing
    RLNS_7::Union{Float64,Missing} = missing
    RLTS_7::Union{Float64,Missing} = missing
    TAUS_7::Union{Float64,Missing} = missing
    VPAR_7::Union{Float64,Missing} = missing
    VPAR_SHEAR_7::Union{Float64,Missing} = missing

    MASS_8::Union{Float64,Missing} = missing
    AS_8::Union{Float64,Missing} = missing
    ZS_8::Union{Float64,Missing} = missing
    RLNS_8::Union{Float64,Missing} = missing
    RLTS_8::Union{Float64,Missing} = missing
    TAUS_8::Union{Float64,Missing} = missing
    VPAR_8::Union{Float64,Missing} = missing
    VPAR_SHEAR_8::Union{Float64,Missing} = missing

    MASS_9::Union{Float64,Missing} = missing
    AS_9::Union{Float64,Missing} = missing
    ZS_9::Union{Float64,Missing} = missing
    RLNS_9::Union{Float64,Missing} = missing
    RLTS_9::Union{Float64,Missing} = missing
    TAUS_9::Union{Float64,Missing} = missing
    VPAR_9::Union{Float64,Missing} = missing
    VPAR_SHEAR_9::Union{Float64,Missing} = missing

    MASS_10::Union{Float64,Missing} = missing
    AS_10::Union{Float64,Missing} = missing
    ZS_10::Union{Float64,Missing} = missing
    RLNS_10::Union{Float64,Missing} = missing
    RLTS_10::Union{Float64,Missing} = missing
    TAUS_10::Union{Float64,Missing} = missing
    VPAR_10::Union{Float64,Missing} = missing
    VPAR_SHEAR_10::Union{Float64,Missing} = missing

    AS_2::Union{Float64,Missing} = missing
    AS_3::Union{Float64,Missing} = missing
    BETAE::Union{Float64,Missing} = missing
    DEBYE::Union{Float64,Missing} = missing
    DELTA_LOC::Union{Float64,Missing} = missing
    DRMAJDX_LOC::Union{Float64,Missing} = missing
    KAPPA_LOC::Union{Float64,Missing} = missing
    P_PRIME_LOC::Union{Float64,Missing} = missing
    Q_LOC::Union{Float64,Missing} = missing
    Q_PRIME_LOC::Union{Float64,Missing} = missing
    RLNS_1::Union{Float64,Missing} = missing
    RLNS_2::Union{Float64,Missing} = missing
    RLNS_3::Union{Float64,Missing} = missing
    RLTS_1::Union{Float64,Missing} = missing
    RLTS_2::Union{Float64,Missing} = missing
    RMAJ_LOC::Union{Float64,Missing} = missing
    RMIN_LOC::Union{Float64,Missing} = missing
    S_KAPPA_LOC::Union{Float64,Missing} = missing
    TAUS_2::Union{Float64,Missing} = missing
    VEXB_SHEAR::Union{Float64,Missing} = missing
    VPAR_1::Union{Float64,Missing} = missing
    VPAR_SHEAR_1::Union{Float64,Missing} = missing
    XNUE::Union{Float64,Missing} = missing
    ZEFF::Union{Float64,Missing} = missing

    # switches
    UNITS::Union{String,Missing} = missing
    ALPHA_ZF::Union{Float64,Missing} = missing
    USE_MHD_RULE::Union{Bool,Missing} = missing
    NKY::Union{Int,Missing} = missing
    SAT_RULE::Union{Int,Missing} = missing
    KYGRID_MODEL::Union{Int,Missing} = missing
    NMODES::Union{Int,Missing} = missing
    NBASIS_MIN::Union{Int,Missing} = missing
    NBASIS_MAX::Union{Int,Missing} = missing
    XNU_MODEL::Union{Int,Missing} = missing
    USE_AVE_ION_GRID::Union{Bool,Missing} = missing
    ALPHA_QUENCH::Union{Int,Missing} = missing
    ALPHA_MACH::Union{Float64,Missing} = missing
    WDIA_TRAPPED::Union{Float64,Missing} = missing
    USE_BPAR::Union{Bool,Missing} = missing
    USE_BPER::Union{Bool,Missing} = missing

    # other
    THETA_TRAPPED::Float64 = 0.7

    _Qgb::Union{Float64,Missing} = missing
end

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

Base.@kwdef mutable struct InputCGYRO
    EQUILIBRIUM_MODEL::Union{Int,Missing} = missing
    RMIN::Union{Float64,Missing} = missing
    RMAJ::Union{Float64,Missing} = missing
    SHIFT::Union{Float64,Missing} = missing
    KAPPA::Union{Float64,Missing} = missing
    S_KAPPA::Union{Float64,Missing} = missing
    DELTA::Union{Float64,Missing} = missing
    S_DELTA::Union{Float64,Missing} = missing
    ZETA::Union{Float64,Missing} = missing
    S_ZETA::Union{Float64,Missing} = missing
    ZMAG::Union{Float64,Missing} = missing
    DZMAG::Union{Float64,Missing} = missing
    Q::Union{Float64,Missing} = missing
    S::Union{Float64,Missing} = missing
    BTCCW::Union{Float64,Missing} = missing
    IPCCW::Union{Float64,Missing} = missing
    UDSYMMETRY_FLAG::Union{Float64,Missing} = missing

    SHAPE_COS0::Union{Float64,Missing} = missing
    SHAPE_S_COS0::Union{Float64,Missing} = missing
    SHAPE_COS1::Union{Float64,Missing} = missing
    SHAPE_S_COS1::Union{Float64,Missing} = missing
    SHAPE_COS2::Union{Float64,Missing} = missing
    SHAPE_S_COS2::Union{Float64,Missing} = missing
    SHAPE_COS3::Union{Float64,Missing} = missing
    SHAPE_S_COS3::Union{Float64,Missing} = missing
    SHAPE_SIN3::Union{Float64,Missing} = missing
    SHAPE_S_SIN3::Union{Float64,Missing} = missing

    PROFILE_MODEL::Union{Int,Missing} = missing
    QUASINEUTRAL_FLAG::Union{Int,Missing} = missing
    NONLINEAR_FLAG::Union{Int,Missing} = missing
    ZF_TEST_MODE::Union{Int,Missing} = missing
    SILENT_FLAG::Union{Int,Missing} = missing
    AMP::Union{Float64,Missing} = missing
    AMP0::Union{Float64,Missing} = missing

    N_FIELD::Union{Int,Missing} = missing
    BETAE_UNIT::Union{Float64,Missing} = missing
    BETAE_UNIT_SCALE::Union{Float64,Missing} = missing
    BETA_STAR_SCALE::Union{Float64,Missing} = missing
    LAMBDA_DEBYE::Union{Float64,Missing} = missing
    LAMBDA_DEBYE_SCALE::Union{Float64,Missing} = missing

    N_RADIAL::Union{Int,Missing} = missing
    BOX_SIZE::Union{Int,Missing} = missing
    N_TOROIDAL::Union{Int,Missing} = missing
    KY::Union{Float64,Missing} = missing
    N_THETA::Union{Int,Missing} = missing
    N_XI::Union{Int,Missing} = missing
    N_ENERGY::Union{Int,Missing} = missing
    E_MAX::Union{Float64,Missing} = missing

    UP_RADIAL::Union{Float64,Missing} = missing
    UP_THETA::Union{Float64,Missing} = missing
    UP_ALPHA::Union{Float64,Missing} = missing
    NUP_RADIAL::Union{Int,Missing} = missing
    NUP_THETA::Union{Int,Missing} = missing
    NUP_ALPHA::Union{Int,Missing} = missing
    VELOCITY_ORDER::Union{Int,Missing} = missing
    UPWIND_SINGLE_FLAG::Union{Int,Missing} = missing

    DELTA_T_METHOD::Union{Int,Missing} = missing
    DELTA_T::Union{Float64,Missing} = missing
    ERROR_TOL::Union{Float64,Missing} = missing
    MAX_TIME::Union{Float64,Missing} = missing
    FREQ_TOL::Union{Float64,Missing} = missing
    PRINT_STEP::Union{Float64,Missing} = missing
    RESTART_STEP::Union{Float64,Missing} = missing

    N_SPECIES::Union{Int,Missing} = missing
    Z_1::Union{Float64,Missing} = missing
    MASS_1::Union{Float64,Missing} = missing
    DENS_1::Union{Float64,Missing} = missing
    TEMP_1::Union{Float64,Missing} = missing
    DLNNDR_1::Union{Float64,Missing} = missing
    DLNTDR_1::Union{Float64,Missing} = missing

    Z_2::Union{Float64,Missing} = missing
    MASS_2::Union{Float64,Missing} = missing
    DENS_2::Union{Float64,Missing} = missing
    TEMP_2::Union{Float64,Missing} = missing
    DLNNDR_2::Union{Float64,Missing} = missing
    DLNTDR_2::Union{Float64,Missing} = missing

    Z_3::Union{Float64,Missing} = missing
    MASS_3::Union{Float64,Missing} = missing
    DENS_3::Union{Float64,Missing} = missing
    TEMP_3::Union{Float64,Missing} = missing
    DLNNDR_3::Union{Float64,Missing} = missing
    DLNTDR_3::Union{Float64,Missing} = missing

    Z_4::Union{Float64,Missing} = missing
    MASS_4::Union{Float64,Missing} = missing
    DENS_4::Union{Float64,Missing} = missing
    TEMP_4::Union{Float64,Missing} = missing
    DLNNDR_4::Union{Float64,Missing} = missing
    DLNTDR_4::Union{Float64,Missing} = missing

    Z_5::Union{Float64,Missing} = missing
    MASS_5::Union{Float64,Missing} = missing
    DENS_5::Union{Float64,Missing} = missing
    TEMP_5::Union{Float64,Missing} = missing
    DLNNDR_5::Union{Float64,Missing} = missing
    DLNTDR_5::Union{Float64,Missing} = missing

    Z_6::Union{Float64,Missing} = missing
    MASS_6::Union{Float64,Missing} = missing
    DENS_6::Union{Float64,Missing} = missing
    TEMP_6::Union{Float64,Missing} = missing
    DLNNDR_6::Union{Float64,Missing} = missing
    DLNTDR_6::Union{Float64,Missing} = missing

    Z_7::Union{Float64,Missing} = missing
    MASS_7::Union{Float64,Missing} = missing
    DENS_7::Union{Float64,Missing} = missing
    TEMP_7::Union{Float64,Missing} = missing
    DLNNDR_7::Union{Float64,Missing} = missing
    DLNTDR_7::Union{Float64,Missing} = missing

    Z_8::Union{Float64,Missing} = missing
    MASS_8::Union{Float64,Missing} = missing
    DENS_8::Union{Float64,Missing} = missing
    TEMP_8::Union{Float64,Missing} = missing
    DLNNDR_8::Union{Float64,Missing} = missing
    DLNTDR_8::Union{Float64,Missing} = missing

    Z_9::Union{Float64,Missing} = missing
    MASS_9::Union{Float64,Missing} = missing
    DENS_9::Union{Float64,Missing} = missing
    TEMP_9::Union{Float64,Missing} = missing
    DLNNDR_9::Union{Float64,Missing} = missing
    DLNTDR_9::Union{Float64,Missing} = missing

    Z_10::Union{Float64,Missing} = missing
    MASS_10::Union{Float64,Missing} = missing
    DENS_10::Union{Float64,Missing} = missing
    TEMP_10::Union{Float64,Missing} = missing
    DLNNDR_10::Union{Float64,Missing} = missing
    DLNTDR_10::Union{Float64,Missing} = missing

    Z_11::Union{Float64,Missing} = missing
    MASS_11::Union{Float64,Missing} = missing
    DENS_11::Union{Float64,Missing} = missing
    TEMP_11::Union{Float64,Missing} = missing
    DLNNDR_11::Union{Float64,Missing} = missing
    DLNTDR_11::Union{Float64,Missing} = missing

    NU_EE::Union{Float64,Missing} = missing
    COLLISION_MODEL::Union{Int64,Missing} = missing
    COLLISION_FIELD_MODEL::Union{Int64,Missing} = missing
    COLLISION_MOM_RESTORE::Union{Int64,Missing} = missing
    COLLISION_ENE_RESTORE::Union{Int64,Missing} = missing
    COLLISION_ENE_DIFFUSION::Union{Int64,Missing} = missing
    COLLISION_KPERP::Union{Int64,Missing} = missing
    COLLISION_PRECISION_MODE::Union{Int64,Missing} = missing
    GPU_BIGMEM_FLAG::Union{Int64,Missing} = missing

    ROTATION_MODEL::Union{Int64,Missing} = missing
    GAMMA_E::Union{Float64,Missing} = missing
    GAMMA_P::Union{Float64,Missing} = missing
    MACH::Union{Float64,Missing} = missing
    GAMMA_E_SCALE::Union{Float64,Missing} = missing
    GAMMA_P_SCALE::Union{Float64,Missing} = missing
    MACH_SCALE::Union{Float64,Missing} = missing

    N_GLOBAL::Union{Int64,Missing} = missing
    NU_GLOBAL::Union{Float64,Missing} = missing
    FIELD_PRINT_FLAG::Union{Int64,Missing} = missing
    MOMENT_PRINT_FLAG::Union{Int64,Missing} = missing
    GFLUX_PRINT_FLAG::Union{Int64,Missing} = missing
    H_PRINT_FLAG::Union{Int64,Missing} = missing

end

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
    input_tglf = InputTGLFs([TGLFNN.InputTGLF() for k in eachindex(gridpoint_cp)])

    signb = sign(Bt)
    input_tglf.SIGN_BT = signb
    input_tglf.SIGN_IT = @. signb * sign(q)

    input_tglf.NS = length(ions) + 1 # add 1 to include electrons

    # electrons first for TGLF
    input_tglf.MASS_1 = me / md
    input_tglf.TAUS_1 = 1.0
    input_tglf.AS_1 = 1.0
    input_tglf.ZS_1 = -1.0
    input_tglf.RLNS_1 = @. a * dlnnedr
    input_tglf.RLTS_1 = @. a * dlntedr

    c_s = GACODE.c_s.(Te)
    w0 = @. -cp1d.rotation_frequency_tor_sonic
    w0p = IMAS.gradient(rmin, w0)
    gamma_p = @. @views -Rmaj[gridpoint_cp] * w0p[gridpoint_cp]
    gamma_e = @. @views -rmin[gridpoint_cp] / q * w0p[gridpoint_cp]
    mach = @. @views Rmaj[gridpoint_cp] * w0[gridpoint_cp] / c_s
    input_tglf.VPAR_1 = @. -input_tglf.SIGN_IT * mach
    input_tglf.VPAR_SHEAR_1 = @. -input_tglf.SIGN_IT * (a / c_s) * gamma_p
    input_tglf.VEXB_SHEAR = @. -gamma_e * (a / c_s)

    for iion in eachindex(ions)
        species = iion + 1
        Ti_full = ions[iion].temperature
        dlntidr_full = .-IMAS.calc_z(rmin, Ti_full, :backward)
        @views Ti = Ti_full[gridpoint_cp]
        @views dlntidr = dlntidr_full[gridpoint_cp]

        Zi = IMAS.avgZ(ions[iion].element[1].z_n, Ti)
        setproperty!(input_tglf, Symbol("ZS_$species"), Zi)
        setproperty!(input_tglf, Symbol("MASS_$species"), ions[iion].element[1].a .* mp ./ md)

        ni_full = ions[iion].density_thermal ./ m³_to_cm³
        dlnnidr_full = .-IMAS.calc_z(rmin, ni_full, :backward)
        @views ni = ni_full[gridpoint_cp]
        @views dlnnidr = dlnnidr_full[gridpoint_cp]

        setproperty!(input_tglf, Symbol("TAUS_$species"), Ti ./ Te)
        setproperty!(input_tglf, Symbol("AS_$species"), ni ./ ne)
        setproperty!(input_tglf, Symbol("VPAR_$species"), input_tglf.VPAR_1)
        setproperty!(input_tglf, Symbol("VPAR_SHEAR_$species"), input_tglf.VPAR_SHEAR_1)
        setproperty!(input_tglf, Symbol("RLNS_$species"), a .* dlnnidr)
        setproperty!(input_tglf, Symbol("RLTS_$species"), a .* dlntidr)
    end

    input_tglf.BETAE = @. 8π * ne * k * Te / bunit^2
    input_tglf.XNUE = @. a / c_s * sqrt(ions[1].element[1].a) * e^4 * π * ne * (24.0 - log(sqrt(ne) / Te)) / (sqrt(me) * (k * Te)^1.5)
    input_tglf.ZEFF = @views cp1d.zeff[gridpoint_cp]
    rho_s = @views GACODE.rho_s(cp1d, eqt)[gridpoint_cp]
    input_tglf.DEBYE = @. 7.43e2 * sqrt(Te / ne) / rho_s
    input_tglf.RMIN_LOC = @. @views rmin[gridpoint_cp] / a
    input_tglf.RMAJ_LOC = @. @views Rmaj[gridpoint_cp] / a
    input_tglf.ZMAJ_LOC = 0
    input_tglf.DRMINDX_LOC = 1.0

    drmaj = IMAS.gradient(rmin, Rmaj)

    input_tglf.DRMAJDX_LOC = @views drmaj[gridpoint_cp]
    input_tglf.DZMAJDX_LOC = 0.0

    input_tglf.Q_LOC = @. abs(q)

    input_tglf.KAPPA_LOC = @views kappa[gridpoint_cp]

    skappa = rmin .* IMAS.gradient(rmin, kappa) ./ kappa
    sdelta = rmin .* IMAS.gradient(rmin, delta)
    szeta = rmin .* IMAS.gradient(rmin, zeta)

    input_tglf.S_KAPPA_LOC = @views skappa[gridpoint_cp]
    input_tglf.DELTA_LOC = @views delta[gridpoint_cp]
    input_tglf.S_DELTA_LOC = @views sdelta[gridpoint_cp]
    input_tglf.ZETA_LOC = @views zeta[gridpoint_cp]
    input_tglf.S_ZETA_LOC = @views szeta[gridpoint_cp]

    press = cp1d.pressure_thermal
    Pa_to_dyn = 10.0

    dpdr = @views IMAS.gradient(rmin, press)[gridpoint_cp] .* Pa_to_dyn
    input_tglf.P_PRIME_LOC = @. @views abs(q) / (rmin[gridpoint_cp] / a)^2 * rmin[gridpoint_cp] / bunit^2 * dpdr

    dqdr = @views IMAS.gradient(rmin, q_profile)[gridpoint_cp]
    input_tglf.Q_PRIME_LOC = @. @views q * a^2 / rmin[gridpoint_cp] * dqdr

    # saturation rules
    input_tglf.ALPHA_ZF = 1.0 # 1 = default, -1 = low ky cutoff kypeak search
    input_tglf.USE_MHD_RULE = false
    input_tglf.NMODES = input_tglf.NS .+ 2 # capture main branches: ES each species + BPER + VPAR_SHEAR
    input_tglf.NKY = 12 # 12 is default, 16 for smoother spectrum
    input_tglf.ALPHA_QUENCH = 0 # 0 = spectral shift, 1 = quench
    input_tglf.SAT_RULE = parse(Int, split(string(sat), "sat")[end])
    if sat == :sat2 || sat == :sat3
        input_tglf.UNITS = "CGYRO"
        input_tglf.KYGRID_MODEL = 4
        input_tglf.NBASIS_MIN = 2
        input_tglf.NBASIS_MAX = 6
        input_tglf.USE_AVE_ION_GRID = true
        input_tglf.XNU_MODEL = 3
        input_tglf.WDIA_TRAPPED = 1.0
    else
        input_tglf.UNITS = "GYRO"
        if sat == :sat1
        elseif sat == :sat1geo
            input_tglf.UNITS = "CGYRO"
        elseif sat == :sat0quench
            input_tglf.ALPHA_QUENCH = 1
        end
        input_tglf.KYGRID_MODEL = 1
        input_tglf.NBASIS_MIN = 2
        input_tglf.NBASIS_MAX = 4
        input_tglf.USE_AVE_ION_GRID = false # default is false
        input_tglf.XNU_MODEL = 2
        input_tglf.WDIA_TRAPPED = 0.0
    end

    # electrostatic/electromagnetic
    if electromagnetic
        input_tglf.USE_BPER = true
        input_tglf.USE_BPAR = true
    else
        input_tglf.USE_BPER = false
        input_tglf.USE_BPAR = false
    end

    input_tglf.ALPHA_MACH = 0.0
    return input_tglf
end

function InputCGYRO(dd::IMAS.dd, gridpoint_cp::Integer, lump_ions::Bool)
    input_cgyro = TGLFNN.InputCGYRO()

    eq = dd.equilibrium
    eqt = eq.time_slice[]
    eqt1d = eqt.profiles_1d
    cp1d = dd.core_profiles.profiles_1d[]

    if lump_ions
        ions = IMAS.lump_ions_as_bulk_and_impurity(cp1d)
    else
        ions = cp1d.ion
    end

    e = IMAS.cgs.e # statcoul
    k = IMAS.cgs.k # erg/eV
    mp = IMAS.cgs.mp # g
    me = IMAS.cgs.me # g
    md = 2 * mp # g
    m_to_cm = IMAS.cgs.m_to_cm
    m³_to_cm³ = IMAS.cgs.m³_to_cm³
    T_to_Gauss = IMAS.cgs.T_to_Gauss

    rmin = GACODE.r_min_core_profiles(eqt1d, cp1d.grid.rho_tor_norm)
    a = rmin[end]

    Rmaj = IMAS.interp1d(eqt1d.rho_tor_norm, m_to_cm * 0.5 * (eqt1d.r_outboard .+ eqt1d.r_inboard)).(cp1d.grid.rho_tor_norm)

    input_cgyro.RMIN = (rmin/a)[gridpoint_cp]
    input_cgyro.RMAJ = Rmaj[gridpoint_cp] / a

    dens_e = cp1d.electrons.density ./ m³_to_cm³
    ne = dens_e[gridpoint_cp]
    dlnnedr = -IMAS.calc_z(rmin ./ a, dens_e, :backward)
    dlnnedr = dlnnedr[gridpoint_cp]

    temp_e = cp1d.electrons.temperature
    Te = temp_e[gridpoint_cp]
    dlntedr = -IMAS.calc_z(rmin ./ a, temp_e, :backward)
    dlntedr = dlntedr[gridpoint_cp]

    n_norm = ne
    t_norm = Te
    for iion in eachindex(ions)
        species = iion

        Ti = ions[iion].temperature ./ t_norm
        dlntidr = -IMAS.calc_z(rmin ./ a, Ti, :backward)
        Ti = Ti[gridpoint_cp]
        dlntidr = dlntidr[gridpoint_cp]

        Zi = IMAS.avgZ(ions[iion].element[1].z_n, Ti * t_norm)
        setproperty!(input_cgyro, Symbol("Z_$species"), Zi)
        setproperty!(input_cgyro, Symbol("MASS_$species"), ions[iion].element[1].a .* mp / md)

        ni = ions[iion].density_thermal ./ m³_to_cm³ / n_norm
        dlnnidr = -IMAS.calc_z(rmin ./ a, ni, :backward)
        ni = ni[gridpoint_cp]
        dlnnidr = dlnnidr[gridpoint_cp]

        setproperty!(input_cgyro, Symbol("TEMP_$species"), Ti)
        setproperty!(input_cgyro, Symbol("DENS_$species"), ni)
        setproperty!(input_cgyro, Symbol("DLNNDR_$species"), dlnnidr)
        setproperty!(input_cgyro, Symbol("DLNTDR_$species"), dlntidr)
    end

    # electrons last for CGYRO
    i = length(ions) + 1
    setproperty!(input_cgyro, Symbol("DENS_$i"), ne / n_norm)
    setproperty!(input_cgyro, Symbol("TEMP_$i"), Te / t_norm)
    setproperty!(input_cgyro, Symbol("MASS_$i"), me / md)
    setproperty!(input_cgyro, Symbol("Z_$i"), -1.0)
    setproperty!(input_cgyro, Symbol("DLNNDR_$i"), dlnnedr)
    setproperty!(input_cgyro, Symbol("DLNTDR_$i"), dlntedr)

    input_cgyro.N_SPECIES = length(ions) + 1 # add 1 to include electrons

    c_s = GACODE.c_s(cp1d)[gridpoint_cp]
    loglam = 24.0 - log(sqrt(ne) / (Te))
    nu_ee = (a / c_s) * (loglam * 4 * pi * ne * e^4) / ((2 * k * Te)^(3 / 2) * me^(1 / 2))
    input_cgyro.NU_EE = nu_ee

    kappa = IMAS.interp1d(eqt1d.rho_tor_norm, eqt1d.elongation).(cp1d.grid.rho_tor_norm)
    input_cgyro.KAPPA = kappa[gridpoint_cp]

    skappa = rmin .* IMAS.gradient(rmin, kappa) ./ kappa
    input_cgyro.S_KAPPA = skappa[gridpoint_cp]

    drmaj = IMAS.gradient(rmin, Rmaj)
    input_cgyro.SHIFT = drmaj[gridpoint_cp]

    delta = IMAS.interp1d(eqt1d.rho_tor_norm, 0.5 * (eqt1d.triangularity_lower + eqt1d.triangularity_upper)).(cp1d.grid.rho_tor_norm)
    input_cgyro.DELTA = delta[gridpoint_cp]
    sdelta = rmin .* IMAS.gradient(rmin, delta)
    input_cgyro.S_DELTA = sdelta[gridpoint_cp]

    zeta =
        IMAS.interp1d(
            eqt1d.rho_tor_norm,
            0.25 * (eqt1d.squareness_lower_inner .+ eqt1d.squareness_lower_outer .+ eqt1d.squareness_upper_inner .+ eqt1d.squareness_upper_outer)
        ).(cp1d.grid.rho_tor_norm)
    input_cgyro.ZETA = zeta[gridpoint_cp]
    szeta = rmin .* IMAS.gradient(rmin, zeta)
    input_cgyro.S_ZETA = szeta[gridpoint_cp]

    Z0 = IMAS.interp1d(eqt1d.rho_tor_norm, eqt1d.geometric_axis.z * 1e2).(cp1d.grid.rho_tor_norm)
    input_cgyro.ZMAG = Z0[gridpoint_cp] / a
    sZ0 = IMAS.gradient(rmin, Z0)
    input_cgyro.DZMAG = sZ0[gridpoint_cp]

    q_profile = IMAS.interp1d(eqt1d.rho_tor_norm, eqt1d.q).(cp1d.grid.rho_tor_norm)
    q = q_profile[gridpoint_cp]

    input_cgyro.Q = q

    w0 = -1 * cp1d.rotation_frequency_tor_sonic
    w0p = IMAS.gradient(rmin, w0)
    gamma_p = -Rmaj[gridpoint_cp] * w0p[gridpoint_cp]
    gamma_e = -rmin[gridpoint_cp] / q * w0p[gridpoint_cp]
    mach = Rmaj[gridpoint_cp] * w0[gridpoint_cp] / c_s
    input_cgyro.GAMMA_P = (a / c_s) * gamma_p
    input_cgyro.GAMMA_E = (a / c_s) * gamma_e
    input_cgyro.MACH = mach

    bunit = IMAS.interp1d(eqt1d.rho_tor_norm, GACODE.bunit(eqt1d) .* T_to_Gauss).(cp1d.grid.rho_tor_norm[gridpoint_cp])

    input_cgyro.BETAE_UNIT = 8.0 * pi * ne * k * Te / bunit^2

    dqdr = IMAS.gradient(rmin, q_profile)[gridpoint_cp]
    s = rmin[gridpoint_cp] / q * dqdr
    input_cgyro.S = s

    Bt = eqt.global_quantities.vacuum_toroidal_field.b0

    input_cgyro.BTCCW = sign(Bt)
    input_cgyro.IPCCW = sign(Bt) * sign(q)

    return input_cgyro
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
        println("input.tglf file type detected")
        for line in lines
            field, value = map(strip, split(line, "="))
            ip_dict[Symbol(field)] = value
        end
    elseif !all(line -> contains(line, "="), lines)
        println("input.tglf.gen file type detected")
        for line in lines
            value, field = map(strip, split(line, "  "))
            ip_dict[Symbol(field)] = value
        end
    else
        error("invalid input.tglf or input.tglf.gen file")
    end

    field_types = fieldtypes(TGLFNN.InputTGLF)
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
    load(filename::AbstractString)

Reads filename (`input.tglf` or `input.tglf.gen` format) and returns input_tglf::InputTGLF
"""
function load(filename::String)
    return load(InputTGLF(), filename)
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
    return collect(asyncmap(input_tglf -> TGLFNN.run_tglf(input_tglf), input_tglfs))
end

export run_tglf

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
    compare_two_input_tglfs(itp_1::TGLFNN.InputTGLF, itp_2::TGLFNN.InputTGLF)

Compares two input_tglfs, prints the difference and stores the difference in a new InputTGLF
"""
function compare_two_input_tglfs(itp_1::TGLFNN.InputTGLF, itp_2::TGLFNN.InputTGLF)
    itp_diff = TGLFNN.InputTGLF()
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
