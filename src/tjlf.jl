function InputTJLF(input_tglf::InputTGLF)
    nky = TJLF.get_ky_spectrum_size(input_tglf.NKY, input_tglf.KYGRID_MODEL)
    input_tjlf = InputTJLF{Float64}(input_tglf.NS, nky)
    input_tjlf.WIDTH_SPECTRUM .= 1.65
    input_tjlf.FIND_WIDTH = true # first case should find the widths
    return update_input_tjlf!(input_tjlf, input_tglf)
end

"""
    update_input_tjlf!(input_tglf::InputTGLF)

Modifies an InputTJLF from a InputTGLF
"""
function update_input_tjlf!(input_tjlf::InputTJLF, input_tglf::TGLFNN.InputTGLF)
    input_tjlf.NWIDTH = 21

    for fieldname in intersect(fieldnames(typeof(input_tglf)), fieldnames(typeof(input_tjlf)))
        setfield!(input_tjlf, fieldname, getfield(input_tglf, fieldname))
    end

    for i in 1:input_tglf.NS
        input_tjlf.ZS[i] = getfield(input_tglf, Symbol("ZS_", i))
        input_tjlf.AS[i] = getfield(input_tglf, Symbol("AS_", i))
        input_tjlf.MASS[i] = getfield(input_tglf, Symbol("MASS_", i))
        input_tjlf.RLNS[i] = getfield(input_tglf, Symbol("RLNS_", i))
        input_tjlf.RLTS[i] = getfield(input_tglf, Symbol("RLTS_", i))
        input_tjlf.TAUS[i] = getfield(input_tglf, Symbol("TAUS_", i))
        input_tjlf.VPAR[i] = getfield(input_tglf, Symbol("VPAR_", i))
        input_tjlf.VPAR_SHEAR[i] = getfield(input_tglf, Symbol("VPAR_SHEAR_", i))
    end

    # Defaults
    input_tjlf.KY = 0.3
    input_tjlf.ALPHA_E = 1.0
    input_tjlf.ALPHA_P = 1.0
    input_tjlf.XNU_FACTOR = 1.0
    input_tjlf.DEBYE_FACTOR = 1.0
    input_tjlf.RLNP_CUTOFF = 18.0
    input_tjlf.WIDTH = 1.65
    input_tjlf.WIDTH_MIN = 0.3
    input_tjlf.BETA_LOC = 0.0
    input_tjlf.KX0_LOC = 1.0
    input_tjlf.PARK = 1.0
    input_tjlf.GHAT = 1.0
    input_tjlf.GCHAT = 1.0
    input_tjlf.WD_ZERO = 0.1
    input_tjlf.LINSKER_FACTOR = 0.0
    input_tjlf.GRADB_FACTOR = 0.0
    input_tjlf.FILTER = 2.0
    input_tjlf.THETA_TRAPPED = 0.7
    input_tjlf.ETG_FACTOR = 1.25
    input_tjlf.DAMP_PSI = 0.0
    input_tjlf.DAMP_SIG = 0.0

    input_tjlf.FIND_EIGEN = true
    input_tjlf.NXGRID = 16

    input_tjlf.ADIABATIC_ELEC = false
    input_tjlf.VPAR_MODEL = 0
    input_tjlf.NEW_EIKONAL = true
    input_tjlf.USE_BISECTION = true
    input_tjlf.USE_INBOARD_DETRAPPED = false
    input_tjlf.IFLUX = true
    input_tjlf.IBRANCH = -1
    input_tjlf.KX0_LOC = 0.0
    input_tjlf.ALPHA_ZF = -1

    # check converison
    TJLF.checkInput(input_tjlf)

    return input_tjlf
end

function run_tjlf(input_tjlf::InputTJLF)
    QL_flux_out = TJLF.run_tjlf(input_tjlf)
    return GACODE.FluxSolution(TJLF.Qe(QL_flux_out), TJLF.Qi(QL_flux_out), TJLF.Γe(QL_flux_out), TJLF.Γi(QL_flux_out), TJLF.Πi(QL_flux_out))
end

function run_tjlf(input_tglf::InputTGLF)
    input_tjlf = InputTJLF(input_tglf)
    return run_tjlf(input_tjlf)
end