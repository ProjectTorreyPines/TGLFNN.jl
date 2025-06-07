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