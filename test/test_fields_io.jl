# ==============================================================================
# test/test_fields_io.jl — gate IO: save_fields / load_fields round-trip, all three solvers.
#
# WHY IT EXISTS (2026-09-14). `src/fields_io.jl` is the package's one output format —
# README.md §4 documents its schema and promises three things:
#
#   (1) the fields come back exactly as the solver returned them,
#   (2) the file also carries WHAT PRODUCED THEM — the resolved term switches, the
#       `show_equations` printout, every scalar knob, the git revision,
#   (3) it is written in PLAIN TYPES, so it reads back without FiVo loaded.
#
# None of that was exercised anywhere: at the time this file was written `save_fields`
# and `load_fields` had zero callers in the package, zero in `phd-git`, and appeared in
# no test and no example. A documented I/O format that nothing reads is a format that
# breaks the first time a field is added to a model — silently, in someone's archive.
# Claim (3) is the one that cannot be checked in-process, so part (d) re-opens the file
# in a SUBPROCESS that loads only JLD2 and never touches FiVo.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_fields_io.jl
# ==============================================================================

using Printf
using Test
using JLD2

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main.jl"))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro
using .hydro2d
const H  = hydro
const H2 = hydro2d

const TMPDIR = mktempdir()

# A charge-free conformal EOS keeps the cold-start recovery out of the α ≲ −20 corner
# (README.md, known limitations); this gate is about the file, not about the physics.
eos1d() = H.ConformalHQEOS(m_hq = 1.5, g_hq = 6.0)

# ---------------------------------------------------------------- 1+1D bulk
function make_1d()
    g = H.make_grid_1d(48; rmax = 8.0)
    m = H.build_model_1d(; eos = eos1d(), enable_shear = true, eta_over_s = 0.1,
                           tauShear_coeff = 0.2, enable_diff = true, kappa_coeff = 0.1163,
                           consistent_fm = true, terms = (fm_inertial = false,))
    U = H.allocate_state(g, m)
    H.initialize_from_radial!(U, g, m, 0.6, r -> 0.20 + 0.20exp(-r^2/8), r -> -4.0)
    res = H.run_sim_1d!(U, g, m; τ0 = 0.6, τfinal = 0.8)
    res.ok || error("1-D seed run failed at τ = $(res.τ)")
    return (m, H.fields_1d(g, U, m; τ = res.τ, work = res.work))
end

# ---------------------------------------------------------------- 2+1D
function make_2d()
    g = H2.make_grid2d(24, 24; xmax = 8.0, ymax = 8.0)
    m = H2.build_model_2d(; eos = H2.ConformalHQEOS(m_hq = 1.5, g_hq = 6.0),
                            enable_shear = true, eta_over_s = 0.1,
                            enable_diff = true, kappa_coeff = 0.1163,
                            consistent_fm = true)
    U = H2.allocate_state(g, m)
    H2.initialize_from_radial!(U, g, m, 0.6, r -> 0.20 + 0.20exp(-r^2/8), r -> -4.0)
    res = H2.run_sim_2d!(U, g, m; τ0 = 0.6, τfinal = 0.8)
    res.ok || error("2-D seed run failed at τ = $(res.τ)")
    return (m, H2.fields_2d(g, U, res.work, m))
end

"""Every array of a NamedTuple of arrays, compared bit for bit (`isequal`, so NaN == NaN)."""
function same_fields(a::NamedTuple, b)
    keys(a) == keys(b) || return false
    all(isequal(getfield(a, k), getfield(b, k)) for k in keys(a))
end

function main()
    @testset "IO — save_fields / load_fields" begin

        # ---------------------------------------------------------- (a) 1+1D bulk
        m1, f1 = make_1d()
        p1 = joinpath(TMPDIR, "run1d.jld2")
        H.save_fields(p1, f1; model = m1, meta = (event = 3, note = "gate"))
        d1 = H.load_fields(p1)

        @printf("  (a) 1+1D: %d fields, %d cells, file %.1f kB\n",
                length(keys(f1)), length(f1.T), filesize(p1)/1024)
        @test same_fields(f1, d1.fields)                     # (1) the fields come back exactly
        @test d1.solver == "FiVo 1+1D bulk"
        @test d1.terms == H.terms_namedtuple(m1.terms)       # (2) … with the terms that made them
        @test d1.terms.fm_inertial == false                  #     the switch this run actually set
        @test d1.terms.nu_gradalpha == true
        @test occursin("consistent_fm", d1.model) || occursin("first moment", d1.model)
        @test d1.model_fields["kappa_coeff"] == repr(0.1163)
        @test d1.model_fields["consistent_fm"] == repr(true)
        @test haskey(d1.model_fields, "eos")
        @test d1.meta["event"] == 3 && d1.meta["note"] == "gate"
        @test !isempty(d1.created) && !isempty(d1.fivo_revision)

        # ---------------------------------------------------------- (b) 2+1D
        m2, f2 = make_2d()
        p2 = joinpath(TMPDIR, "run2d.jld2")
        H2.save_fields(p2, f2; model = m2)
        d2 = H2.load_fields(p2)

        @printf("  (b) 2+1D: %d fields, %dx%d cells, file %.1f kB\n",
                length(keys(f2)), size(f2.T)..., filesize(p2)/1024)
        @test same_fields(f2, d2.fields)
        @test d2.solver == "FiVo 2+1D"                       # the label must not collapse onto 1-D
        @test d2.terms == H2.terms_namedtuple(m2.terms)   # NB: hydro2d.Terms is its own type
        @test d2.model_fields["consistent_fm"] == repr(true)
        @test d2.fields.T isa AbstractMatrix                 # 2-D stays 2-D through the file

        # ---------------------------------------------------------- (c) an IS2-shaped result
        # `run_static_IS2_test` returns a Dict and carries its own "terms"; save_fields must
        # pick those up and label the file without a model (README.md §4).
        is2 = Dict{String,Any}("r_grid" => collect(0.0:0.5:4.0), "t_grid" => [0.6, 0.8],
                               "n" => rand(9, 2), "nur" => rand(9, 2),
                               "terms" => H.resolve_terms(H.without(:acceleration)),
                               "consistent_fm" => true, "consistent_m2" => true)
        p3 = joinpath(TMPDIR, "is2.jld2")
        H.save_fields(p3, is2)
        d3 = H.load_fields(p3)
        @printf("  (c) IS2 Dict: solver=%s, terms recorded=%s\n", d3.solver, d3.terms !== nothing)
        @test d3.solver == "FiVo 1+1D charm IS2"
        @test d3.terms !== nothing
        @test d3.terms.fm_inertial == false                  # `without(:acceleration)` reached the file
        @test d3.fields["n"] == is2["n"]

        # ------------------------------------------- (d) it reads WITHOUT FiVo loaded
        # The documented promise is "plain types only". Checked in a subprocess that
        # `using JLD2` and nothing else: if any value needed a FiVo type to reconstruct,
        # JLD2 hands back a reconstructed placeholder and the type test below fails.
        probe = joinpath(TMPDIR, "probe.jl")
        write(probe, """
            # NOTE: everything is inside a function on purpose — an `ok` accumulated in a
            # top-level `for` (or in a `do` block) is a new local, which is how a PASS/FAIL
            # check silently passes here (CLAUDE.md, trap 1).
            using JLD2
            # An IS2 result is a Dict and carries its own `terms` NamedTuple beside the arrays,
            # so a NamedTuple of Bools is a legal leaf here too.
            plain(v) = (v isa AbstractArray && eltype(v) <: Union{Number,Bool}) ||
                       v isa Number || v isa AbstractString ||
                       (v isa NamedTuple && all(plain, values(v)))
            function check(path)
                jldopen(path, "r") do f
                    fl = f["fields"]
                    (fl isa NamedTuple || fl isa AbstractDict) || return false
                    all(plain, values(fl)) || return false
                    t = f["terms"]
                    (t === nothing || (t isa NamedTuple && all(x -> x isa Bool, values(t)))) || return false
                    f["model"] isa AbstractString || return false
                    f["model_fields"] isa AbstractDict || return false
                    f["solver"] isa AbstractString || return false
                    return true
                end
            end
            exit(all(check, ARGS) ? 0 : 1)
            """)
        JLBIN = joinpath(Sys.BINDIR, Base.julia_exename())
        plain = success(run(ignorestatus(`$JLBIN --project=$_ROOT $probe $p1 $p2 $p3`)))
        println("  (d) re-read in a process with only JLD2 loaded: ", plain ? "plain types" : "NOT plain")
        @test plain
    end
end
main()
