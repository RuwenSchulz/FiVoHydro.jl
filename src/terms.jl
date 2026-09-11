# ==============================================================================
# src/terms.jl — ONE register of the switchable terms of FiVo's equations,
# shared by the 1+1D solvers (main.jl, main2IS2.jl) and the 2+1D solver
# (main2D.jl). Dimension-agnostic: no grid, no state, only names and Bools.
#
# THE INTERFACE, in one screen
# ----------------------------
# Every solver takes a `terms` keyword and accepts the same four spellings:
#
#     terms = :default                      # the equations as they stand
#     terms = :homogeneous                  # a homogeneous medium at rest (see below)
#     terms = without(:vorticity, :acceleration)      # drop by physical ingredient
#     terms = (preset = :full, without = (:acceleration,), fm_dlnh = false)
#     terms = (with = (:vorticity,),)       # the default, plus both vorticity couplings
#
# `show_terms()` prints the register; each solver's `show_equations(model)` prints
# the equations it integrates with every term marked [x] / [ ].
#
# PRESETS
#   :default      what the code integrates when nothing is said: every derived term
#                 ON except the two vorticity couplings (a decision, not a default
#                 of physics — see `m2_vorticity`, `shear_vorticity`).
#   :full         every term ON, the vorticity couplings included.
#   :homogeneous  the reduction to a HOMOGENEOUS MEDIUM AT REST: every term built
#                 from a gradient or a rate of the medium — ∇T, a = Du, ∇u (shear,
#                 expansion, vorticity), DT — is dropped. What survives is what the
#                 charm's OWN gradients drive: ∇α (first moment), ∇ν and Dα (second
#                 moment). This is exactly the frame the SHIPPED first-moment row was
#                 derived in, so with `consistent_fm = true` it reproduces
#                 `consistent_fm = false` to the last bit (gate: test_terms1d.jl T3,
#                 test_terms2d.jl Gt7). ⚠ θ carries the Bjorken 1/τ, so "homogeneous"
#                 here also drops the longitudinal dilution — as the shipped row does.
#   :none         every term OFF: each field relaxes to zero on its own clock,
#                 τ Dν + ν = 0 etc. A referee state (pure exponential decay).
#
# INGREDIENTS (what `without(...)` and the NamedTuple keys `without` / `with` take)
#   :fugacity_gradient   ∇^⟨μ⟩α       :temperature_gradient  ∇^⟨μ⟩T
#   :acceleration        a^μ = Du^μ   (the INERTIAL terms; alias :inertia)
#   :shear  σ^{μν}   :expansion  θ   :vorticity  ω^{μν}
#   :velocity_gradient   ∇u — the whole gradient: implies :shear, :expansion, :vorticity
#   :cooling             DT           :fugacity_rate   Dα
#   :current_gradient    ∇ν           (charm second moment only)
#   :medium_gradients    group: everything a homogeneous static medium lacks
#                        (= the :homogeneous preset, relative to any base)
# A term is removed by `without(x)` if it is built from ANY ingredient x implies.
#
# SECTORS AND REFUSAL
# Each term belongs to a sector — a model flag (`enable_diff`, `consistent_fm`,
# `consistent_m2`, `enable_shear`). A term NAMED explicitly is refused when its
# sector is off (an attribution run must not silently measure nothing — the rule
# test_terms2d.jl Gt5 enforces). A change that comes from a PRESET or an
# INGREDIENT, in a disabled sector, is inert and is reset to the default instead
# of refused: `terms = :homogeneous` means "whatever is enabled, homogeneously".
#
# RADIAL SYMMETRY (1+1D)
# Three terms vanish IDENTICALLY in the components the 1+1D solvers evolve:
#   m2_vorticity, shear_vorticity   — a radial flow has no vorticity;
#   m2_projector                    — the 1-D second moment lives in the parallel-
#                                     transported triad (l, φ̂, η̂), whose channels are
#                                     orthogonal to u, so −τ(u^μc^ν + u^νc^μ) has no
#                                     component there (hq_consistent_m2.jl header).
# The 1-D solvers accept either value for them and say "≡ 0 in 1+1D".
#
# BIT-IDENTITY. Every switch is read as `on ? term : 0.0` in the order the
# expression had before the switch existed, and `:default` is the struct's own
# defaults, so an unswitched run is bit-identical to the code before this file.
#
# History: the 2-D solver had these seventeen switches as `Terms2D`
# (src2d/terms2d.jl, 2026-09-10); on 2026-09-11 they moved here, gained ingredient
# tags, presets and `shear_vorticity`, and were wired into the 1+1D solvers.
# `Terms2D` remains as an alias.
# ==============================================================================

"""
    Terms(; kwargs...)

Per-term switches of FiVo's equations (1+1D and 2+1D alike). Every field is a
`Bool`; `true` means the term is carried. Build one with `resolve_terms`, or pass
any spec (`:homogeneous`, `without(:vorticity)`, a `NamedTuple`) as a solver's
`terms` keyword. Field list, sectors and ingredients: `show_terms()`.
"""
Base.@kwdef struct Terms
    # ── charm first moment: the current ν^μ ────────────────────────────────────
    nu_gradalpha::Bool   = true   # −κ ∇^⟨i⟩α                     the shipped fugacity drive
    fm_gradT::Bool       = true   # (D_s/T)(n + T∂n/∂T) ∇^⟨i⟩T     pressure gradient, T channel
    fm_inertial::Bool    = true   # τ_n n a^i                     inertial
    fm_nu_gradu::Bool    = true   # τ_n ν^m ∇_m u^i               current riding the flow gradient
    fm_expansion::Bool   = true   # τ_n θ ν^i                     expansion (incl. Bjorken dilution)
    fm_dlnh::Bool        = true   # (D_s/T) h′ DT ν^i             transport of the coefficient h
    # ── charm second moment: π_Q^{μν} and its trace Π_Q ────────────────────────
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
    # ── the medium's shear stress ──────────────────────────────────────────────
    shear_vorticity::Bool = false # 2τ_π π^{λ⟨μ}ω_λ^{ν⟩}  the medium's vorticity coupling — OFF by default
end

"""
    TERM_REGISTER

`(name, sector, ingredients, what)` for every `Terms` field, in field order.
`sector` is the model flag the term belongs to; `ingredients` are what it is built
from (what `without` matches). `show_terms`, the refusal in `check_terms` and the
term gates iterate over this, so a new field that is not listed here fails
test_terms2d.jl Gt1 and test_terms1d.jl T1.
"""
const TERM_REGISTER = (
    (:nu_gradalpha,    :enable_diff,   (:fugacity_gradient,),           "−κ ∇^⟨i⟩α                          fugacity drive (the shipped row)"),
    (:fm_gradT,        :consistent_fm, (:temperature_gradient,),        "−(D_s/T)(n + T∂n/∂T) ∇^⟨i⟩T         pressure gradient, T channel"),
    (:fm_inertial,     :consistent_fm, (:acceleration,),                "−τ_n n a^i                          inertial"),
    (:fm_nu_gradu,     :consistent_fm, (:velocity_gradient,),           "−τ_n ν^m ∇_m u^i                    current riding the flow gradient"),
    (:fm_expansion,    :consistent_fm, (:expansion,),                   "−τ_n θ ν^i                          expansion (carries the Bjorken 1/τ)"),
    (:fm_dlnh,         :consistent_fm, (:cooling,),                     "−(D_s/T) h′ DT ν^i                  transport of h = m K₃/K₂"),
    (:m2_nu_gradient,  :consistent_m2, (:current_gradient,),            "2η_Q σ_(ν)^{ij}   |  ζ_Q θ_(ν)            (i)   gradient of the current"),
    (:m2_bg_gradu,     :consistent_m2, (:shear, :expansion),            "2η̄ σ^{ij}         |  (5/3) η̄ θ            (ii)  gradient of the medium"),
    (:m2_bg_DlnT,      :consistent_m2, (:cooling,),                     "                  |  η̄ (A/B) D ln T       (ii)  cooling"),
    (:m2_bg_Dalpha,    :consistent_m2, (:fugacity_rate,),               "                  |  η̄ Dα                 (ii)  fugacity rate"),
    (:m2_expansion,    :consistent_m2, (:expansion, :cooling),          "τ_M((5/3)θ + D ln C) π_Q^{ij} | … Π_Q       (iii) expansion"),
    (:m2_pi_sigma,     :consistent_m2, (:shear,),                       "2τ_M π_Q^{λ⟨i}σ^{j⟩}_λ | (2/3)τ_M π_Q:σ     (iii) shear coupling"),
    (:m2_PiQ_sigma,    :consistent_m2, (:shear,),                       "2τ_M Π_Q σ^{ij}   |                       (iii) δM-extension"),
    (:m2_vorticity,    :consistent_m2, (:vorticity,),                   "2τ_M π_Q^{λ⟨i}ω_λ^{j⟩} |                    (iii) vorticity coupling"),
    (:m2_projector,    :consistent_m2, (:acceleration,),                "−τ_M(u^i c^j + u^j c^i) |                   the Δ-projector on Dπ_Q, c^j = π_Q^{jβ}a_β"),
    (:m2_accel_nu,     :consistent_m2, (:acceleration,),                "2λ_a a^⟨i ν^j⟩    |  (D_s A/3T) a·ν        (iv)  acceleration × current"),
    (:m2_nu_gradTh,    :consistent_m2, (:temperature_gradient,),        "(D_s/T) ν^⟨i∇^j⟩(Th) | (5D_s/6T) ν·∇(Th)  (iv)  enthalpy gradient × current"),
    (:shear_vorticity, :enable_shear,  (:vorticity,),                   "2τ_π π^{λ⟨i}ω_λ^{j⟩}                 medium shear: vorticity coupling"),
)

"""Terms that vanish identically in the components the 1+1D solvers evolve (see header)."""
const ZERO_IN_RADIAL = (:m2_vorticity, :m2_projector, :shear_vorticity)

const INGREDIENTS = (:fugacity_gradient, :temperature_gradient, :acceleration,
                     :shear, :expansion, :vorticity, :velocity_gradient,
                     :cooling, :fugacity_rate, :current_gradient)

const INGREDIENT_GROUPS = (
    inertia           = (:acceleration,),
    velocity_gradient = (:velocity_gradient, :shear, :expansion, :vorticity),
    medium_gradients  = (:temperature_gradient, :acceleration, :velocity_gradient,
                         :shear, :expansion, :vorticity, :cooling),
)

const TERM_PRESETS = (:default, :full, :homogeneous, :none)

"""
    expand_ingredients(xs) -> Vector{Symbol}

Resolve group names (`:inertia`, `:velocity_gradient`, `:medium_gradients`) to the
ingredients they stand for. An unknown name is an error listing the valid ones.
"""
function expand_ingredients(xs)
    out = Symbol[]
    for x in xs
        x isa Symbol || error("Terms: ingredient must be a Symbol, got $(repr(x))")
        if haskey(INGREDIENT_GROUPS, x)
            append!(out, INGREDIENT_GROUPS[x])
        elseif x in INGREDIENTS
            push!(out, x)
        else
            error("Terms: unknown ingredient :$x. Valid: " *
                  join(string.(":", INGREDIENTS), ", ") * "; groups: " *
                  join(string.(":", keys(INGREDIENT_GROUPS)), ", "))
        end
    end
    return unique(out)
end

"""True if the register entry `name` is built from any of `ingr` (already expanded)."""
function _term_uses(name::Symbol, ingr)
    for (n, _, ing, _) in TERM_REGISTER
        n === name && return any(in(ingr), ing)
    end
    error("Terms: no register entry for $name")
end

"""
    terms_preset(p::Symbol) -> Terms

The preset as a `Terms`, before any sector is consulted. See the header.
"""
function terms_preset(p::Symbol)
    p === :default && return Terms()
    names = fieldnames(Terms)
    if p === :full
        return Terms(; (n => true for n in names)...)
    elseif p === :none
        return Terms(; (n => false for n in names)...)
    elseif p === :homogeneous
        ingr = expand_ingredients((:medium_gradients,))
        return Terms(; (n => (getfield(Terms(), n) && !_term_uses(n, ingr)) for n in names)...)
    end
    error("Terms: unknown preset :$p. Valid: " * join(string.(":", TERM_PRESETS), ", "))
end

"""
    without(ingredients...; from = :default) -> NamedTuple spec

Drop every term built from any of `ingredients` (see `show_terms()`), starting from
the preset `from`. Pass the result as a solver's `terms`:

    terms = without(:vorticity, :acceleration)
    terms = without(:acceleration; from = :full)
"""
without(xs::Symbol...; from::Symbol = :default) = (preset = from, without = xs)

"""
    resolve_terms(spec; sector_on = _ -> true) -> Terms

Turn any accepted spelling of `terms` into a `Terms`:

* a `Terms` — returned as is (every field counts as EXPLICIT);
* a `Symbol` — a preset (`:default`, `:full`, `:homogeneous`, `:none`);
* a `NamedTuple` — optional `preset`, `without`, `with`, and any `Terms` field
  names, applied in that order (explicit names last, so they win).

`sector_on(sector::Symbol)::Bool` tells which sectors the model carries. A change
that came from a preset or an ingredient, in a disabled sector, is reset to the
default (it is inert there); an EXPLICIT name in a disabled sector is kept, so
that `check_terms` refuses it. Unknown names are errors.
"""
resolve_terms(t::Terms; sector_on = _ -> true) = t
resolve_terms(p::Symbol; sector_on = _ -> true) = resolve_terms((preset = p,); sector_on)
function resolve_terms(nt::NamedTuple; sector_on = _ -> true)
    valid = fieldnames(Terms)
    ctrl  = (:preset, :without, :with)
    bad = [k for k in keys(nt) if !(k in valid) && !(k in ctrl)]
    isempty(bad) || error("Terms: unknown term(s) $(bad). Valid names: " *
                          join(string.(valid), ", ") * " — plus `preset`, `without`, `with`.")
    base = terms_preset(get(nt, :preset, :default))
    vals = Dict{Symbol,Bool}(n => getfield(base, n) for n in valid)
    if haskey(nt, :without)
        ingr = expand_ingredients(_as_tuple(nt.without))
        for n in valid
            _term_uses(n, ingr) && (vals[n] = false)
        end
    end
    if haskey(nt, :with)
        ingr = expand_ingredients(_as_tuple(nt.with))
        for n in valid
            _term_uses(n, ingr) && (vals[n] = true)
        end
    end
    # implicit changes in a disabled sector are inert: reset them to the default
    def = Terms()
    for (n, sector, _, _) in TERM_REGISTER
        sector_on(sector) || (vals[n] = getfield(def, n))
    end
    # explicit names last, kept even in a disabled sector (check_terms refuses them)
    for k in keys(nt)
        k in ctrl && continue
        v = nt[k]
        v isa Bool || error("Terms: `$k` must be a Bool, got $(repr(v))")
        vals[k] = v
    end
    return Terms(; (n => vals[n] for n in valid)...)
end
_as_tuple(x::Symbol) = (x,)
_as_tuple(x) = Tuple(x)

"""
    check_terms(terms, sector_on; solver = "FiVo")

Refuse a term switch that cannot act: a term turned OFF (or, for a default-off
term, turned ON) whose sector is disabled. `sector_on(sector)::Bool`.
"""
function check_terms(t::Terms, sector_on; solver::AbstractString = "FiVo")
    def = Terms()
    for (name, sector, _, what) in TERM_REGISTER
        getfield(t, name) == getfield(def, name) && continue
        sector_on(sector) && continue
        error("$solver: `$name = $(getfield(t, name))` changes a term of the `$sector` sector, " *
              "but `$sector = false`, so the switch would act on nothing. Term: $what. " *
              "Enable `$sector`, or leave `$name` at its default ($(getfield(def, name))).")
    end
    return nothing
end

"""
    active_terms(terms, sector_on) -> Vector{Symbol}

The terms that are both switched on and in an enabled sector — what a run with
this model actually integrates (radial-zero terms included; the caller decides
whether to flag them).
"""
active_terms(t::Terms, sector_on) =
    [n for (n, s, _, _) in TERM_REGISTER if getfield(t, n) && sector_on(s)]

"""
    show_terms([io]; terms = Terms(), sector_on = _ -> true)

Print the register: every term, its sector, its ingredients, and whether `terms`
carries it. With no arguments, the defaults.
"""
show_terms(; kw...) = show_terms(stdout; kw...)
function show_terms(io::IO; terms = Terms(), sector_on = s -> true)
    t = resolve_terms(terms; sector_on)
    println(io, "FiVo terms — `terms = …` accepts a preset ", join(string.(":", TERM_PRESETS), " / "),
            ", without(ingredients...), or a NamedTuple (preset / without / with / term names)")
    for (n, s, ing, what) in TERM_REGISTER
        on = getfield(t, n) && sector_on(s)
        println(io, "  ", on ? "[x]" : "[ ]", " ", rpad(string(n), 16), " ", rpad(string(s), 14), " ",
                rpad(join(string.(":", ing), " "), 40), " ", strip(what))
    end
    println(io, "  ingredients: ", join(string.(":", INGREDIENTS), " "),
            "   groups: ", join(("$(k) = " * join(string.(":", v), "+") for (k, v) in pairs(INGREDIENT_GROUPS)), ", "))
    return nothing
end
