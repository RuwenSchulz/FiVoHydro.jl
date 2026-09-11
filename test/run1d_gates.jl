# ==============================================================================
# test/run1d_gates.jl — run the 1+1D validation ladder (bulk solver + charm IS2).
#
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run1d_gates.jl
#   FIVO1D_TIER=fast …                                   # the ~20 s tier Pkg.test() runs
#
# The 1-D twin of run2d_gates.jl, built 2026-09-11. Until then the 1-D solver's time
# evolution was covered by nothing beyond a few steps in Pkg.test() (FiVoBenchmark
# was wired into no gate), and three defects lived under that: SSPRK3 was first order,
# π was zeroed every other step on charge-free states at rest, and ∂_τu^r came out
# at ~4 % of its value in every relaxation. Each gate below would have caught one.
#
# Every gate is a closed-form or semi-analytic REFEREE (test/analytic_referees.jl),
# not a copy of the code it checks. Each runs in its own process (several include
# main.jl, main2IS2.jl and main2D.jl together); the runner exits 1 on any failure,
# and a listed gate whose file is missing is a failure, not a skip.
# ==============================================================================

using Printf

const HERE = @__DIR__
const PROJ = normpath(joinpath(HERE, ".."))
const TIER = lowercase(get(ENV, "FIVO1D_TIER", "full"))
TIER in ("fast", "full") || error("FIVO1D_TIER must be \"fast\" or \"full\", got \"$TIER\"")

#            name                                                file                        fast?
const GATES = [
    ("T   term switches (bulk + IS2), Euler cancellation",   "test_terms1d.jl",          true),
    ("A4  viscous Gubser vs semi-analytic ODE",              "test_gubser_viscous1d.jl", true),
    ("A1  ideal Bjorken: order, charge, EOS ODE",            "test_bjorken1d.jl",        false),
    ("X1  diffusion mode: 1D, IS2, 2D vs one referee",       "test_diffusion_mode.jl",   false),
    ("    IS2 drive relaxes onto Navier-Stokes",             "test_is2_drive.jl",        false),
    ("    IS2 characteristic speeds",                        "test_is2_causality.jl",    false),
    ("    density-frame flux + conservation",                "test_density_frame_flux.jl", false),
    ("    BDNK causal: exact Milne ODE + telegraph",         "test_bdnk_causal.jl",      false),
    ("    M1 moment ladder",                                 "test_m1_gates.jl",         false),
]

function main()
    nfail = 0; nrun = 0
    println("="^74)
    println("FiVo 1+1D validation ladder" * (TIER == "fast" ? "  [FAST TIER]" : ""))
    println("="^74)
    for (name, file, in_fast) in GATES
        TIER == "fast" && !in_fast && continue
        nrun += 1
        path = joinpath(HERE, file)
        if !isfile(path)
            @printf("  %-54s  %s\n", name, "FAIL (file missing: $file)")
            nfail += 1
            continue
        end
        t0 = time()
        io = IOBuffer()
        ok = try
            run(pipeline(`$(Base.julia_cmd()) -t auto --project=$PROJ $path`; stdout = io, stderr = io))
            true
        catch
            false
        end
        ok || (nfail += 1)
        @printf("  %-54s  %s  (%.0f s)\n", name, ok ? "PASS" : "FAIL", time() - t0)
        ok || println(String(take!(io)))
    end
    println("="^74)
    @printf("%d/%d gates passed  (tier: %s)\n", nrun - nfail, nrun, TIER)
    exit(nfail == 0 ? 0 : 1)
end
main()
