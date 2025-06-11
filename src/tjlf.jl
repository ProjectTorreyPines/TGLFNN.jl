function run_tjlf(input_tjlf::InputTJLF)
    QL_flux_out = TJLF.run_tjlf(input_tjlf)
    return GACODE.FluxSolution(TJLF.Qe(QL_flux_out), TJLF.Qi(QL_flux_out), TJLF.Γe(QL_flux_out), TJLF.Γi(QL_flux_out), TJLF.Πi(QL_flux_out))
end

function run_tjlf(input_tglf::InputTGLF)
    input_tjlf = InputTJLF{Float64}(input_tglf)
    return run_tjlf(input_tjlf)
end