# ==============================================================================
# test/run2d_gates.jl — run the whole 2+1D validation ladder.
#
#   julia -t auto --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/run2d_gates.jl
#
# Each gate runs in its own subprocess: several of them include main.jl-side files
# and main2D.jl, whose modules would otherwise collide in one session. Exits 1 if
# any gate fails, so this is usable as a CI step.
#
# The ladder and what each gate is for: TWOD_PROGRAM.md §6.
# ==============================================================================

using Printf

const HERE = @__DIR__
const PROJ = normpath(joinpath(HERE, ".."))

# FIVO2D_TIER=fast runs only the gates marked `true` below: algebra, primitive recovery and the
# charm closures, ~30 s in total (measured 2026-09-10, six gates). Three of them take a few
# short solves — Gc5/Gm5 (the flags are inert when off), Gc7 (the first moment's SIGN on a
# Bjorken solve, which no algebraic gate can see) and Gt6 (the term switches act on a solve) —
# so "no time evolution", which this said until 2026-09-10, is no longer true of the tier; it
# still exercises no fluxes on a physical IC and no regulators. That is the tier CI runs on every push
# (.github/workflows/ci.yml) — the full ladder does real 2-D solves and takes tens of minutes,
# which is why it stayed out of CI entirely until 2026-09-08. `fast` is a smoke test, NOT the
# validation ladder: it cannot see anything the timestepper, the fluxes or the regulators do.
# Run the full ladder by hand after touching src2d/ or main2D.jl, and read TWOD_PROGRAM.md §6.
const TIER = lowercase(get(ENV, "FIVO2D_TIER", "full"))
TIER in ("fast", "full") || error("FIVO2D_TIER must be \"fast\" or \"full\", got \"$TIER\"")

#            name                                 file                        in fast tier?
const GATES = [
    ("shear closure algebra",              "test_shear2d_algebra.jl",         true),
    ("primitive recovery",                 "test_primrec2d.jl",               true),
    ("recovery vs 1-D on production locus","test_primrec2d_vs_1d.jl",         true),
    ("Gc  consistent first moment",        "test_consistent_fm2d.jl",         true),
    ("Gm  consistent second moment",       "test_consistent_m22d.jl",         true),
    ("Gt  per-term switches + vorticity",  "test_terms2d.jl",                 true),
    ("G0  Bjorken",                        "test_bjorken2d.jl", false),
    ("G0b Bjorken + nonlinear bulk",       "test_bjorken_bulk2d.jl", false),
    ("G1  Gubser (analytic 2-D)",           "test_gubser2d.jl", false),
    ("G1v Gubser viscous (semi-analytic)",  "test_gubser_viscous2d.jl", false),
    ("Gs  sound speed + attenuation",      "test_sound2d.jl", false),
    ("G2  shear + bulk",                   "test_dissipation2d.jl", false),
    ("G3  charge / diffusion",             "test_charge2d.jl", false),
    ("G3g charge advection on Gubser",     "test_charge_gubser2d.jl", false),
    ("Gk  charge dispersion + k_*",         "test_charge_dispersion2d.jl", false),
    ("G4  reproduction vs 1-D production", "test_reproduction2d.jl", false),
    ("G5  production IC, all sectors",      "test_production_allsectors2d.jl", false),
    ("G6  non-axisymmetric IC",             "test_elliptic2d.jl", false),
    ("G7  dissipative sectors vs 1-D",      "test_dissipative_vs_1d.jl", false),
    ("G8  un-averaged production IC",       "test_unaveraged_ic2d.jl", false),
    ("G9  single-event (fluctuating) IC",   "test_fluctuating_ic2d.jl", false),
]

function main()
    nfail = 0
    nrun  = 0
    println("="^74)
    println("FiVo 2+1D validation ladder" * (TIER == "fast" ? "  [FAST TIER — smoke only]" : ""))
    TIER == "fast" && println("⚠ algebra + recovery only; no time evolution is exercised. " *
                              "Run FIVO2D_TIER=full before trusting src2d/ changes.")
    println("="^74)
    for (name, file, in_fast) in GATES
        TIER == "fast" && !in_fast && continue
        nrun += 1
        path = joinpath(HERE, file)
        if !isfile(path)
            # A listed gate whose file is gone is a FAILURE, not a skip: this script is the only
            # thing that runs the 2-D ladder, so a silent skip is how a gate stops existing without
            # anyone noticing. (Before 2026-09-08 this printed SKIP and still counted as passed.)
            @printf("  %-38s  %s\n", name, "FAIL (file missing: $file)")
            nfail += 1
            continue
        end
        io = IOBuffer()
        ok = try
            run(pipeline(`$(Base.julia_cmd()) -t auto --project=$PROJ $path`;
                         stdout = io, stderr = io))
            true
        catch
            false
        end
        out = String(take!(io))
        @printf("  %-38s  %s\n", name, ok ? "PASS" : "FAIL")
        for ln in split(out, '\n')
            # echo the measurement lines, which are what the gates are actually for
            (startswith(ln, "  ") && !startswith(ln, "   ")) && !isempty(strip(ln)) &&
                println("      ", strip(ln))
        end
        ok || (nfail += 1; println(out))
    end
    println("="^74)
    @printf("%d/%d gates passed  (tier: %s)\n", nrun - nfail, nrun, TIER)
    exit(nfail == 0 ? 0 : 1)
end

main()
