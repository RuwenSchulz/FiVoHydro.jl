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
#   * TERM switches inside the charm closures, which had none: the consistent
#     first moment was all-or-nothing, and so was every one of the ~ten terms of
#     the consistent second moment. Those live in `Terms2D`, one Bool per term,
#     carried by the model as `model.terms`.
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
#                          terms = (fm_inertial = false,))
#     show_equations(m)
#
# The equations themselves, term by term with their provenance and gates:
# EQUATIONS2D.md.
# ==============================================================================

"""
    Terms2D(; kwargs...)

Per-term switches for the charm sector of the 2+1D solver. Every field is a
`Bool`; `true` means the term is carried. Pass to `build_model_2d` either as a
`Terms2D` or as a `NamedTuple` of the fields you want to change:

    build_model_2d(; ..., terms = (fm_inertial = false, m2_vorticity = true))

A switch whose sector is off (`fm_*` without `consistent_fm`, `m2_*` without
`consistent_m2`, `nu_gradalpha` without `enable_diff`) is refused rather than
ignored, so an attribution run cannot silently measure nothing.

Field list and the term each controls: `TERMS_2D`, or `show_equations(model)`.
"""
Base.@kwdef struct Terms2D
    # ── first moment: the charge current ν^i ─────────────────────────────────
    nu_gradalpha::Bool   = true   # −κ ∇^⟨i⟩α                     the shipped fugacity drive
    fm_gradT::Bool       = true   # (D_s/T)(n + T∂n/∂T) ∇^⟨i⟩T     pressure gradient, T channel
    fm_inertial::Bool    = true   # τ_n n a^i                     inertial
    fm_nu_gradu::Bool    = true   # τ_n ν^m ∇_m u^i               current riding the flow gradient
    fm_expansion::Bool   = true   # τ_n θ ν^i                     expansion (incl. Bjorken dilution)
    fm_dlnh::Bool        = true   # (D_s/T) h′ DT ν^i             transport of the coefficient h
    # ── second moment: the charm stress π_Q^{ij} and its trace Π_Q ─────────────
    m2_nu_gradient::Bool = true   # 2η_Q σ_(ν) ; ζ_Q θ_(ν)                   class (i)
    m2_bg_gradu::Bool    = true   # 2η̄ σ ; (5/3) η̄ θ                          class (ii)
    m2_bg_DlnT::Bool     = true   # η̄ (A/B) D ln T                   (trace)  class (ii)
    m2_bg_Dalpha::Bool   = true   # η̄ Dα                             (trace)  class (ii)
    m2_expansion::Bool   = true   # τ_M((5/3)θ + D ln C) π_Q ; … Π_Q          class (iii)
    m2_pi_sigma::Bool    = true   # 2τ_M π_Q^{λ⟨μ}σ^{ν⟩}_λ ; (2/3)τ_M π_Q:σ   class (iii)
    m2_PiQ_sigma::Bool   = true   # 2τ_M Π_Q σ                                class (iii), δM-extension
    m2_vorticity::Bool   = false  # 2τ_M π_Q^{λ⟨μ}ω_λ^{ν⟩}                   class (iii) — OFF by default
    m2_projector::Bool   = true   # −τ_M(u^i c^j + u^j c^i), c^j = π_Q^{jβ}a_β  the Δ-projector on Dπ_Q
    m2_accel_nu::Bool    = true   # 2λ_a a^⟨μ ν^ν⟩ ; (D_s A/3T) a·ν           class (iv)
    m2_nu_gradTh::Bool   = true   # (D_s/T) ν^⟨μ∇^ν⟩(Th) ; (5D_s/6T) ν·∇(Th)  class (iv)
end

"""
    TERMS_2D

`(name, sector, what)` for every `Terms2D` field, in field order. `sector` is the
model flag the term belongs to. This is what `show_equations`, the refusal in
`check_terms_2d` and the test of the switches (`test/test_terms2d.jl`) iterate
over, so a new field that is not listed here fails that test.
"""
const TERMS_2D = (
    (:nu_gradalpha,   :enable_diff,   "−κ ∇^⟨i⟩α                          fugacity drive (the shipped row)"),
    (:fm_gradT,       :consistent_fm, "−(D_s/T)(n + T∂n/∂T) ∇^⟨i⟩T         pressure gradient, T channel"),
    (:fm_inertial,    :consistent_fm, "−τ_n n a^i                          inertial"),
    (:fm_nu_gradu,    :consistent_fm, "−τ_n ν^m ∇_m u^i                    current riding the flow gradient"),
    (:fm_expansion,   :consistent_fm, "−τ_n θ ν^i                          expansion (carries the Bjorken 1/τ)"),
    (:fm_dlnh,        :consistent_fm, "−(D_s/T) h′ DT ν^i                  transport of h = m K₃/K₂"),
    (:m2_nu_gradient, :consistent_m2, "2η_Q σ_(ν)^{ij}   |  ζ_Q θ_(ν)            (i)   gradient of the current"),
    (:m2_bg_gradu,    :consistent_m2, "2η̄ σ^{ij}         |  (5/3) η̄ θ            (ii)  gradient of the medium"),
    (:m2_bg_DlnT,     :consistent_m2, "                  |  η̄ (A/B) D ln T       (ii)  cooling"),
    (:m2_bg_Dalpha,   :consistent_m2, "                  |  η̄ Dα                 (ii)  fugacity rate"),
    (:m2_expansion,   :consistent_m2, "τ_M((5/3)θ + D ln C) π_Q^{ij} | … Π_Q       (iii) expansion"),
    (:m2_pi_sigma,    :consistent_m2, "2τ_M π_Q^{λ⟨i}σ^{j⟩}_λ | (2/3)τ_M π_Q:σ     (iii) shear coupling"),
    (:m2_PiQ_sigma,   :consistent_m2, "2τ_M Π_Q σ^{ij}   |                       (iii) δM-extension"),
    (:m2_vorticity,   :consistent_m2, "2τ_M π_Q^{λ⟨i}ω_λ^{j⟩} |                    (iii) vorticity coupling"),
    (:m2_projector,   :consistent_m2, "−τ_M(u^i c^j + u^j c^i) |                   the Δ-projector on Dπ_Q, c^j = π_Q^{jβ}a_β"),
    (:m2_accel_nu,    :consistent_m2, "2λ_a a^⟨i ν^j⟩    |  (D_s A/3T) a·ν        (iv)  acceleration × current"),
    (:m2_nu_gradTh,   :consistent_m2, "(D_s/T) ν^⟨i∇^j⟩(Th) | (5D_s/6T) ν·∇(Th)  (iv)  enthalpy gradient × current"),
)

"""
    terms2d(x) -> Terms2D

Normalise what a caller passes as `terms`: a `Terms2D` is returned as is, a
`NamedTuple` overrides the defaults field by field. An unknown name is an error
that lists the valid ones (a typo must not silently leave a term on).
"""
terms2d(t::Terms2D) = t
function terms2d(nt::NamedTuple)
    valid = fieldnames(Terms2D)
    bad = [k for k in keys(nt) if !(k in valid)]
    isempty(bad) || error("Terms2D: unknown term(s) $(bad). Valid names: " *
                          join(string.(valid), ", "))
    return Terms2D(; nt...)
end

"""
    check_terms_2d(model)

Refuse a term switch that cannot act: a term turned OFF (or, for the
default-off `m2_vorticity`, turned ON) whose sector is disabled. Called from
`reject_unwired_knobs_2d`, i.e. from `build_model_2d` and at `run_sim_2d!`
entry.
"""
function check_terms_2d(m)
    t = m.terms
    def = Terms2D()
    for (name, sector, what) in TERMS_2D
        getfield(t, name) == getfield(def, name) && continue
        getfield(m, sector) && continue
        error("Terms2D: `$name = $(getfield(t, name))` changes a term of the `$sector` sector, " *
              "but `$sector = false`, so the switch would act on nothing. Term: $what. " *
              "Enable `$sector`, or leave `$name` at its default ($(getfield(def, name))).")
    end
    return nothing
end

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

    println(io, "  shear  ", _mark(m.enable_shear), "  τ_π Δ^{ij}_{αβ} Dπ^{αβ} + π^{ij} = −2η σ^{ij} − δ_ππ θ π^{ij}")
    if m.enable_shear
        @printf(io, "               %s δ_ππ θ π^{ij}, δ_ππ = %.4g τ_π           deltaShear_factor\n",
                _mark(m.deltaShear_factor != 0), m.deltaShear_factor)
        println(io, "               ", _mark(m.shear_projected_deriv),
                " τ_π(u^i π^{jβ} + u^j π^{iβ}) a_β  (projector)  shear_projected_deriv")
        println(io, "               ", _mark(m.relax_advect_pi),
                " τ_π u^k ∂_k π^{ij}  (transverse advection)  relax_advect_pi")
        println(io, "               η = (η/s)·s, τ_π = η/(C_s T s):  η/s = ", _shear_label(m.shear))
    end
    println(io, "  bulk   ", _mark(m.enable_bulk), "  τ_Π DΠ + Π = −ζ θ")
    if m.enable_bulk
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
