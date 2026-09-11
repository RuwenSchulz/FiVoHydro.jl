# ==============================================================================
# src2d/terms2d.jl — switch individual terms of the 2+1D equations on and off,
# and print the equations a model actually integrates.
#
# Two kinds of switch exist in the 2-D model, and this file is the one place
# that lists both:
#
#   * SECTOR and NUMERICS knobs that are fields of `IdealDiffVisc2DModel`
#     (`enable_shear`, `deltaShear_factor = 0` to drop δ_ππ, `shear_projected_deriv`,
#     `relax_advect_nu`, …). They predate this file and every gate uses them, so
#     they stay where they are — a second knob for the same term would be a trap.
#
#   * TERM switches inside the closures, which had none: the consistent
#     first moment was all-or-nothing, and so was every one of the ~ten terms of
#     the consistent second moment. Those live in `Terms` (src/terms.jl, shared
#     with the 1+1D solvers since 2026-09-11; `Terms2D` is its alias here), one
#     Bool per term, carried by the model as `model.terms`. That file also has
#     the presets (`:homogeneous`, `:full`, `:none`) and `without(ingredients…)`.
#
# Defaults reproduce the equations as they stood when the switches were added
# (2026-09-10), bit for bit — the switch is read as `on ? term : 0.0`, so with
# every switch on the arithmetic is the same expression in the same order. Two
# fields are not of that kind:
#   * `m2_vorticity` (default OFF) — a term the derivation contains and neither
#     2-D code carried; off so that turning it on is a decision.
#   * `m2_projector` (default ON) — the Δ-projector on the second moment's
#     comoving derivative, which was MISSING (a bug, not a choice; see
#     hq_consistent_m2_2d.jl). Off reproduces the pre-2026-09-10 rows exactly.
#
# Usage:
#     m = build_model_2d(; enable_diff = true, consistent_fm = true,
#                          terms = (fm_inertial = false,))        # or :homogeneous,
#     show_equations(m)                                           # or without(:acceleration)
#
# The equations themselves, term by term with their provenance and gates:
# EQUATIONS2D.md.
# ==============================================================================

"""
    Terms2D

Alias of the shared `Terms` (src/terms.jl) — the per-term switches, now one type
for the 1+1D and 2+1D solvers. Pass any spec as `build_model_2d(; terms = …)`:

    terms = :homogeneous
    terms = without(:vorticity, :acceleration)
    terms = (fm_inertial = false, m2_vorticity = true)

Field list, sectors and ingredients: `show_terms()`, or `show_equations(model)`.
"""
const Terms2D = Terms

"""
    TERMS_2D

`(name, sector, what)` for every `Terms` field, in field order — `TERM_REGISTER`
without the ingredient column, in the shape `show_equations` and gate Gt1 read.
"""
const TERMS_2D = Tuple((n, s, w) for (n, s, _, w) in TERM_REGISTER)

"""
    sector_on_2d(; enable_diff, consistent_fm, consistent_m2, enable_shear)
        -> (sector::Symbol -> Bool)

Which term sectors a 2-D model will carry, from the flags `build_model_2d`
receives — so `terms` can be resolved before the model exists. (For a built
model the same map is `s -> getfield(model, s)`.)
"""
function sector_on_2d(; enable_diff::Bool = false, consistent_fm::Bool = false,
                        consistent_m2::Bool = false, enable_shear::Bool = false)
    flags = (; enable_diff, consistent_fm, consistent_m2, enable_shear)
    return s -> getfield(flags, s)
end

"""
    terms2d(spec; sector_on = _ -> true) -> Terms

Normalise what a caller passes as `terms` (`resolve_terms`, src/terms.jl): a
`Terms`, a preset `Symbol`, or a `NamedTuple` of `preset` / `without` / `with` /
term names. An unknown name is an error that lists the valid ones (a typo must not
silently leave a term on).
"""
terms2d(x; sector_on = _ -> true) = resolve_terms(x; sector_on)

"""
    check_terms_2d(model)

Refuse a term switch that cannot act: a term turned OFF (or, for a default-off
term, turned ON) whose sector is disabled. Called from `reject_unwired_knobs_2d`,
i.e. from `build_model_2d` and at `run_sim_2d!` entry.
"""
check_terms_2d(m) = check_terms(m.terms, s -> getfield(m, s); solver = "Terms2D")

# ------------------------------------------------------------------------------
# show_equations
# ------------------------------------------------------------------------------

_mark(b::Bool) = b ? "[x]" : "[ ]"

"""
    show_equations([io,] model)

Print the equations `model` integrates — every sector and every switchable term,
marked `[x]` when carried and `[ ]` when not, with the knob that controls it.
Conventions (Milne `(τ,x,y,η)`, `D = u^μ∂_μ`, `θ = ∇_μu^μ`, `a^μ = Du^μ`,
`X^⟨μ⟩ = Δ^μ_ν X^ν`) and the derivations: EQUATIONS2D.md.
"""
show_equations(m) = show_equations(stdout, m)
function show_equations(io::IO, m)
    t = m.terms
    L = m.layout
    println(io, "FiVo 2+1D — the equations this model integrates")
    println(io, "  Milne (τ,x,y,η), g = diag(−1,1,1,τ²), boost invariant; D = u·∂, θ = ∇·u, a = Du")
    println(io)
    println(io, "  medium       ∂_τ(T^{τν}) + ∂_x(T^{xν}) + ∂_y(T^{yν}) = geometric sources   (always)")
    println(io, "               T^{μν} = (e + P + Π) u^μu^ν + (P + Π) g^{μν} + π^{μν},   EoS: ",
            nameof(typeof(m.eos)))
    if L.hasNu
        println(io, "  charge       ∂_τ(τJ^τ) + τ ∂_i J^i = 0,   J^μ = n u^μ + ν^μ")
    end
    println(io)

    println(io, "  shear  ", _mark(m.enable_shear), "  τ_π Δ^{ij}_{αβ} Dπ^{αβ} + π^{ij} = −2η σ^{ij} − δ_ππ θ π^{ij} − …")
    if m.enable_shear
        @printf(io, "               %s δ_ππ θ π^{ij}, δ_ππ = %.4g τ_π           deltaShear_factor\n",
                _mark(m.deltaShear_factor != 0), m.deltaShear_factor)
        println(io, "               ", _mark(m.shear_projected_deriv),
                " τ_π(u^i π^{jβ} + u^j π^{iβ}) a_β  (projector)  shear_projected_deriv")
        println(io, "               ", _mark(m.relax_advect_pi),
                " τ_π u^k ∂_k π^{ij}  (transverse advection)  relax_advect_pi")
        println(io, "               ", _mark(t.shear_vorticity),
                " 2τ_π π^{λ⟨i}ω_λ^{j⟩}  (vorticity coupling, off by default)  terms.shear_vorticity")
        @printf(io, "               %s τ_ππ π^{λ⟨i}σ^{j⟩}_λ, τ_ππ = %.4g τ_π     taupi_pi_factor\n",
                _mark(m.taupi_pi_factor != 0), m.taupi_pi_factor)
        @printf(io, "               %s λ_πΠ Π σ^{ij},       λ_πΠ = %.4g τ_π     lambda_pi_Pi_factor\n",
                _mark(m.lambda_pi_Pi_factor != 0), m.lambda_pi_Pi_factor)
        println(io, "               η = (η/s)·s, τ_π = η/(C_s T s):  η/s = ", _shear_label(m.shear))
    end
    println(io, "  bulk   ", _mark(m.enable_bulk), "  τ_Π DΠ + Π = −ζ θ − δ_ΠΠ θ Π − λ_Ππ π:σ")
    if m.enable_bulk
        @printf(io, "               %s δ_ΠΠ θ Π,  δ_ΠΠ = %.4g τ_Π     deltaPi_factor\n",
                _mark(m.deltaPi_factor != 0), m.deltaPi_factor)
        @printf(io, "               %s λ_Ππ π:σ,  λ_Ππ = %.4g τ_Π     lambda_Pi_pi_factor\n",
                _mark(m.lambda_Pi_pi_factor != 0), m.lambda_Pi_pi_factor)
        println(io, "               ", _mark(m.relax_advect_Pi),
                " τ_Π u^k ∂_k Π  (transverse advection)  relax_advect_Pi")
    end

    println(io, "  charge ", _mark(m.enable_diff), "  τ_n Δ^i_ν Dν^ν + ν^i = drive^i")
    if m.enable_diff
        println(io, "               ", _mark(m.diff_projected_deriv),
                " τ_n u^i (ν·a)  (projector)  diff_projected_deriv")
        println(io, "               ", _mark(m.relax_advect_nu),
                " τ_n u^k ∂_k ν^i  (transverse advection)  relax_advect_nu")
        println(io, "      drive^i =")
        for (name, sector, what) in TERMS_2D
            sector === :enable_diff || sector === :consistent_fm || continue
            on = getfield(t, name) && getfield(m, sector)
            println(io, "               ", _mark(on), " ", what, "   terms.", name,
                    sector === :consistent_fm ? "  (consistent_fm)" : "")
        end
        m.consistent_fm || println(io, "               (consistent_fm = false: only the fugacity drive)")
    end

    if m.consistent_m2 && L.hasM2
        println(io, "  charm second moment [x]   (consistent_m2; passive at c_M = 0)")
        println(io, "      τ_M Δ Dπ_Q^{ij} + π_Q^{ij} + S^{ij} = 0 ,   τ_M DΠ_Q + Π_Q + S_B = 0")
        println(io, "      S^{ij}                |  S_B")
        for (name, sector, what) in TERMS_2D
            sector === :consistent_m2 || continue
            println(io, "               ", _mark(getfield(t, name)), " ", what, "   terms.", name)
        end
        println(io, "               ", _mark(m.relax_advect_m2),
                " τ_M u^k ∂_k (π_Q, Π_Q)  (transverse advection)  relax_advect_m2")
    else
        println(io, "  charm second moment [ ]   (consistent_m2 = false)")
    end
    return nothing
end

_shear_label(sh) = hasproperty(sh, :ηs) ? string(sh.ηs, ", C_s = ", sh.Cs) : "0"
