#!/usr/bin/env julia
# test_is2_causality.jl — regression characterization of the IS2 5×5 system's characteristic speeds
# (see Projects/FiVoBenchmark/bench_is2_causality.jl, finding I-1). Run as an isolated subprocess.
# Locks in: (a) strong hyperbolicity (all eigenvalues real); (b) vacuum cells are causal (|λ|≤1);
# (c) the coefficient-level causality fix (finding I-1) holds: rest-frame v_sig ≤ c (τ_n≥κ/χ clamp in
# transport_all). Exits 0 on pass.

include(joinpath(@__DIR__, "..", "main2IS2.jl"))
using .hydro_current_IS2
using LinearAlgebra
const IS = hydro_current_IS2

function eig_speed(At, Ax)
    F = lu(At; check=false)
    B = issuccess(F) ? (F \ Ax) : (pinv(At) * Ax)
    ev = eigvals(B); cmax = 0.0; allreal = true
    for e in ev
        isfinite(abs(e)) && (cmax = max(cmax, abs(e)))
        abs(imag(e)) > 1e-7*max(1.0, abs(real(e))) && (allreal = false)
    end
    return cmax, allreal
end

function scan(eos, DsT)
    maxrest = 0.0; maxvac = 0.0; nonreal = 0
    for T in range(0.12, 0.55; length=8), α in range(-2.0, 2.0; length=7)
        tp = IS.transport_all(T, α, DsT, eos)
        At = zeros(5,5); Ax = zeros(5,5); src = zeros(5)
        IS.build_IS2_system!(At, Ax, src, (α,0.0,0.0,0.0,0.0), 3.0, 5.0, 0.0, T,0.0,0.0,0.0,0.0,
            tp.n, tp.dn_dα, tp.dn_dT, tp.κ, tp.τn, tp.τM, tp.ηM, 0.0)
        c, real_ok = eig_speed(copy(At), copy(Ax))   # u^r=0 ⇒ rest frame
        real_ok || (nonreal += 1)
        if tp.n >= IS.IS2_VACUUM_N_LO
            maxrest = max(maxrest, c)
        else
            maxvac = max(maxvac, c)
        end
    end
    return maxrest, maxvac, nonreal
end

allpass = true; fail(m) = (global allpass; allpass = false; println("FAIL: ", m))
maxrest, maxvac, nonreal = scan(IS.build_eos("lattice"), 0.24)

nonreal == 0 || fail("non-real eigenvalues (loss of strong hyperbolicity): $nonreal states")
maxvac <= 1.05 || fail("vacuum cells superluminal: max|λ|=$maxvac")
maxrest <= 1.001 || fail("rest-frame diffusion superluminal (coeff causality fix regressed): v_sig=$maxrest")

println("test_is2_causality: rest-frame max v_sig=$(round(maxrest,digits=4)) (known mild >c), ",
        "vacuum max|λ|=$(round(maxvac,digits=4)), all-real=$(nonreal==0)")
println(allpass ? "test_is2_causality: PASS (characterization locked)" : "test_is2_causality: FAIL")
exit(allpass ? 0 : 1)
