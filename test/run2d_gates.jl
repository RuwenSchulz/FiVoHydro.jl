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

const GATES = [
    ("shear closure algebra",              "test_shear2d_algebra.jl"),
    ("primitive recovery",                 "test_primrec2d.jl"),
    ("recovery vs 1-D on production locus","test_primrec2d_vs_1d.jl"),
    ("G0  Bjorken",                        "test_bjorken2d.jl"),
    ("G1  Gubser (analytic 2-D)",           "test_gubser2d.jl"),
    ("G1v Gubser viscous (semi-analytic)",  "test_gubser_viscous2d.jl"),
    ("Gs  sound speed + attenuation",      "test_sound2d.jl"),
    ("G2  shear + bulk",                   "test_dissipation2d.jl"),
    ("G3  charge / diffusion",             "test_charge2d.jl"),
    ("G4  reproduction vs 1-D production", "test_reproduction2d.jl"),
    ("G5  production IC, all sectors",      "test_production_allsectors2d.jl"),
    ("G6  non-axisymmetric IC",             "test_elliptic2d.jl"),
    ("G7  dissipative sectors vs 1-D",      "test_dissipative_vs_1d.jl"),
    ("G8  un-averaged production IC",       "test_unaveraged_ic2d.jl"),
    ("G9  single-event (fluctuating) IC",   "test_fluctuating_ic2d.jl"),
]

function main()
    nfail = 0
    println("="^74)
    println("FiVo 2+1D validation ladder")
    println("="^74)
    for (name, file) in GATES
        path = joinpath(HERE, file)
        if !isfile(path)
            @printf("  %-38s  %s\n", name, "SKIP (not present)")
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
    @printf("%d/%d gates passed\n", length(GATES) - nfail, length(GATES))
    exit(nfail == 0 ? 0 : 1)
end

main()
