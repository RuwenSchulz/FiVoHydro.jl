module hydro_current_M1
# ═════════════════════════════════════════════════════════════════════════════════════════════════
# main2M1.jl — THE CHARM CURRENT AS A MAXIMUM-ENTROPY (LEVERMORE M1) MOMENT SYSTEM.
#
# A PARALLEL module to main2IS2.jl, never a modification of it: production stays byte-identical and
# both solvers can be run on the SAME background for comparison (the CMExperiment pattern).
#
# ── WHY ─────────────────────────────────────────────────────────────────────────────────────────
# `Tex/HeavyQuarkHydro/diag_maxent_closure.jl` + `diag_m1_slab.jl` established, by measurement:
#   • re-taking the first moment of f0·e^δ is the IDENTITY — it cannot regularise ν (2.6e-12);
#   • |ν*_r/n| < 1 is REALIZABILITY of any f ≥ 0, so a solve reporting w ≥ 1 has produced a moment
#     pair no distribution function possesses, and nothing downstream repairs it;
#   • the flux Q = ∫d³p (v cosθ)² f that the ν equation is closed with is, in the kinetic theory,
#     the MaxEnt one to 1.8% (≤11% anywhere in a coupling scan) against 121–370% for the constant
#     closure IS2/Grad uses — a factor 34–67;
#   • and a 3-field MaxEnt system beats both 2-field ones on n AND ν (‖Δν‖₁ 0.038 vs 0.071 vs 0.087).
#
# ── THE SYSTEM ──────────────────────────────────────────────────────────────────────────────────
# The carrier is the boosted Jüttner of blk_spectragallery §A.4 `eq:boostpars`, with T′ promoted
# from `tab:maxent`'s MODELLING CHOICE to a dynamical variable — which is exactly what the third
# field buys.  State: (n′, T′, u′^μ), i.e. an ideal Jüttner gas with its OWN flow and temperature,
#
#     N^μ = n′u′^μ ,      T^{μν} = (e′+P′)u′^μu′^ν − P′g^{μν} ,   e′ = n′ε̄(T′) , P′ = n′T′
#
# and the evolution is exact conservation plus the drag/diffusion transfer:
#
#     ∇_μ N^μ = 0 ,       ∇_μ T^{μν} = S^ν
#
# In Milne (boost-invariant, azimuthally symmetric; √−g = τr, and u′ has only (τ,r) so
# T^{ηη} = P′/τ², T^{φφ} = P′/r²), with conserved densities q_N = τrN^τ, Q_E = τrT^{ττ},
# Q_M = τrT^{τr}:
#
#     ∂_τ q_N + ∂_r(τr N^r)   = 0
#     ∂_τ Q_E + ∂_r(τr T^{rτ}) = τr S^τ − r P′         (the −P′/τ from Γ^τ_{ηη}T^{ηη})
#     ∂_τ Q_M + ∂_r(τr T^{rr}) = τr S^r + τ P′         (the +P′/r from Γ^r_{φφ}T^{φφ})
#
# ⭐ NO TRANSPORT COEFFICIENTS.  There is no τ_n, no τ_M, no c_M, no δ_ππ and no second-moment
# builder: their content is replaced by moments of the reconstructed state.  What used to be a
# 5-component IS2 system with a matrix closure is 3 conservation laws with an algebraic one.
#
# ── 🔑 THE CONE IS STRUCTURAL ───────────────────────────────────────────────────────────────────
# Recovery is the standard ideal-fluid primitive inversion, and it reduces to a SINGLE scalar root
# solve.  With D = N^τ, X = T^{τr}/D, Y = T^{ττ}/D and h(T′) = ε̄(T′) + T′ the enthalpy per particle,
#
#     V = X/√(h²+X²)   identically ,   G(T′) ≡ h² + X² − T′h − Y√(h²+X²) = 0
#
# G is monotone in T′ because h − T′ = ε̄ ≥ M.  |V| < 1 for ANY h > 0, so the charm four-current is
# timelike BY CONSTRUCTION — `_recover_alpha_from_q!`'s J^τ ≤ 0 branch, `IS2_JTAU_NEG`, the
# α = ±200 clamp and `FIVO_IS2_NU_BOUND` all have nothing left to do.  The realizability boundary is
# G(0) = 0, i.e.
#
#     T^{ττ}/N^τ  ≥  √(M² + (T^{τr}/N^τ)²)
#
# — the charm energy per particle may not fall below that of a cold beam carrying its momentum.
# Cells that violate it are COUNTED and reported (`realizability_floor`), never silently clamped.
# And the inversion is conditioned on the RATIOS X, Y, not on n, so the dn/dα → 0 ill-conditioning
# that forces IS2's α clamp in the dilute tail is absent.
#
# ── CONVENTIONS ─────────────────────────────────────────────────────────────────────────────────
# Drag from the PRODUCTION Langevin (`LangevInMedium.jl/src/transport.jl::tau_drag`):
#     1/η_D = M·D_sT/T²  [×ħc for fm]   ⇒   η_D = T²/(M D_sT)/ħc  [fm⁻¹],  κ = 2MTη_D,
# so the equilibrium is the Jüttner e^{−E/T} and the slab validation carries over unchanged.
#
# Entry points mirror main2IS2.jl so the LP1 drop-in can call either:
#     run_static_M1_test(; background_file, DsT, τ0, τfinal, Nr, rmax, ...) -> Dict
#
# ═════════════════════════════════════════════════════════════════════════════════════════════════

using LinearAlgebra
using Printf
using Logging
using JLD2
using SpecialFunctions

const _SRC = joinpath(@__DIR__, "src")
include(joinpath(_SRC, "utils.jl"))
include(joinpath(_SRC, "grid.jl"))
include(joinpath(_SRC, "constants.jl"))
include(joinpath(_SRC, "eos.jl"))
include(joinpath(_SRC, "io.jl"))

export run_static_M1_test, solve_M1, M1Grid1D, M1Background, load_M1_background

# ── knobs (all diagnostic; none is a regulator) ─────────────────────────────────────────────────
# There is deliberately NO vacuum ramp, NO α clamp and NO ν bound here.  If this solver needs one,
# that is a finding about the formulation, not a knob to reach for — see the header.
const M1_TP_LO   = parse(Float64, get(ENV, "FIVO_M1_TP_LO", "1e-5"))   # GeV, T′ bisection floor
const M1_TP_HI   = parse(Float64, get(ENV, "FIVO_M1_TP_HI", "5.0"))    # GeV, T′ bisection ceiling
const M1_N_FLOOR = parse(Float64, get(ENV, "FIVO_M1_N_FLOOR", "1e-300"))
const M1_DSTT_LINEAR = get(ENV, "FIVO_M1_DSTT_LINEAR", "0") == "1"
const M1_DSTT_SLOPE  = parse(Float64, get(ENV, "FIVO_M1_DSTT_SLOPE",  "1.765"))
const M1_DSTT_OFFSET = parse(Float64, get(ENV, "FIVO_M1_DSTT_OFFSET", "-0.159"))
const M1_DSTT_TFO    = parse(Float64, get(ENV, "FIVO_M1_DSTT_TFO",    "0.156"))
const M1_DIAG_TERMS  = get(ENV, "FIVO_M1_DIAG_TERMS", "0") == "1"
_grabdg(v, k, Nr) = (A = zeros(Nr, length(v)); for j in eachindex(v), i in 1:Nr; A[i, j] = v[j][k][i]; end; A)

# ═══════════════════════ Grid / background (same shapes as main2IS2) ═════════════════════════════
struct M1Grid1D
    r::Vector{Float64}
    rF::Vector{Float64}
    dr::Vector{Float64}
end

function M1Grid1D(Nr::Int, rmax::Float64)
    d = rmax/Nr
    M1Grid1D([(i - 0.5)*d for i in 1:Nr], [(i - 1)*d for i in 1:(Nr + 1)], fill(d, Nr))
end

struct M1Background
    r_grid::Vector{Float64}
    t_grid::Vector{Float64}
    T_spl::Any
    ur_spl::Any
    α_spl::Any
    n_spl::Any
    nur_spl::Any
end

function load_M1_background(path::AbstractString)
    isfile(path) || error("Background file not found: $path")
    jldopen(path, "r") do f
        _k(s) = haskey(f, s) ? s : haskey(f, s*"1") ? s*"1" : s
        _o(s) = (haskey(f, s) || haskey(f, s*"1")) ? f[_k(s)] : nothing
        M1Background(Float64.(f["r_grid"]), Float64.(f["t_grid"]), f[_k("T_spline")],
                     _o("ur_spline"), _o("α_spline"), _o("n_spline"), _o("nur_spline"))
    end
end

@inline function _clamp_eval(spl, r, τ, rg, tg)
    spl === nothing && return 0.0
    Float64(spl(clamp(Float64(r), first(rg), last(rg)), clamp(Float64(τ), first(tg), last(tg))))
end
@inline bg_T(bg::M1Background, τ, r)   = _clamp_eval(bg.T_spl,   r, τ, bg.r_grid, bg.t_grid)
@inline bg_ur(bg::M1Background, τ, r)  = _clamp_eval(bg.ur_spl,  r, τ, bg.r_grid, bg.t_grid)
@inline bg_al(bg::M1Background, τ, r)  = _clamp_eval(bg.α_spl,   r, τ, bg.r_grid, bg.t_grid)
@inline bg_n(bg::M1Background, τ, r)   = _clamp_eval(bg.n_spl,   r, τ, bg.r_grid, bg.t_grid)
@inline bg_nur(bg::M1Background, τ, r) = _clamp_eval(bg.nur_spl, r, τ, bg.r_grid, bg.t_grid)
@inline _u_from_ur(ur) = (sqrt(1.0 + Float64(ur)^2), Float64(ur))

# ═══════════════════════ Jüttner scalars ═════════════════════════════════════════════════════════
@inline function _kx(z)
    zc = clamp(Float64(z), 1e-8, 700.0)
    (besselkx(1, zc), besselkx(2, zc))
end
"""Mean energy per particle of a Maxwell-Jüttner at `Tp` [GeV]:  ε̄ = 3T′ + M K₁/K₂."""
@inline function ebar(Tp, M)
    Tc = max(Float64(Tp), M1_TP_LO)
    K1x, K2x = _kx(M/Tc)
    3*Tc + M*K1x/max(K2x, 1e-300)
end
@inline hpp(Tp, M) = ebar(Tp, M) + max(Float64(Tp), M1_TP_LO)

# h is evaluated ~45× per inversion and the inversion runs at every cell AND every face at every RK
# stage, so the Bessel call is tabulated on a log grid spanning T′'s four decades.
const _NH = 8000
const _HT_LO, _HT_HI = M1_TP_LO, M1_TP_HI
const _HTAB = Ref{Vector{Float64}}(Float64[])
const _HSC  = Ref(0.0)
const _HMASS = Ref(0.0)
function _build_h_table!(M)
    (_HMASS[] == M && !isempty(_HTAB[])) && return nothing
    _HTAB[] = [hpp(T, M) for T in exp.(range(log(_HT_LO), log(_HT_HI); length = _NH))]
    _HSC[]  = (_NH - 1)/(log(_HT_HI) - log(_HT_LO))
    _HMASS[] = M
    return nothing
end
@inline function h_fast(Tp)
    t = (log(clamp(Tp, _HT_LO, _HT_HI)) - log(_HT_LO))*_HSC[] + 1
    k = clamp(floor(Int, t), 1, _NH - 1); θ = t - k
    H = _HTAB[]
    (1 - θ)*H[k] + θ*H[k + 1]
end

# ═══════════════════════ Primitive recovery ══════════════════════════════════════════════════════
"""
    recover(D, S, E, M) -> (n′, Tp, V, ok)

`D = N^τ`, `S = T^{τr}`, `E = T^{ττ}` at a cell.  Returns the charm rest-frame density, its own
temperature, its Milne radial velocity `V = u′^r/u′^τ`, and whether the state was realizable.
Single scalar bisection on `G(T′)`; see the header.  `ok = false` means the cell fell below the
cold-beam floor `E/D ≥ √(M² + (S/D)²)` and was placed ON it — counted, never hidden.
"""
@inline function recover(D::Float64, S::Float64, E::Float64, M::Float64)
    D > M1_N_FLOOR || return (0.0, M1_TP_LO, 0.0, true)
    X = S/D; Y = E/D
    G(Tp) = (h = h_fast(Tp); h*h + X*X - Tp*h - Y*sqrt(h*h + X*X))
    if G(M1_TP_LO) > 0                       # below the cold-beam floor: not a moment set of any f≥0
        h = h_fast(M1_TP_LO); R = sqrt(h*h + X*X); V = X/R
        return (D*sqrt(max(1 - V*V, 1e-300)), M1_TP_LO, V, false)
    end
    G(M1_TP_HI) < 0 && begin
        h = h_fast(M1_TP_HI); R = sqrt(h*h + X*X); V = X/R
        return (D*sqrt(max(1 - V*V, 1e-300)), M1_TP_HI, V, false)
    end
    lo, hi = M1_TP_LO, M1_TP_HI
    @inbounds for _ in 1:45                  # geometric bisection: T′ spans four decades
        mid = sqrt(lo*hi)
        G(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    Tp = sqrt(lo*hi)
    h  = h_fast(Tp); V = X/sqrt(h*h + X*X)
    Γ  = 1/sqrt(max(1 - V*V, 1e-300))
    return (D/Γ, Tp, V, true)
end

"""Milne fluxes (N^r, T^{rτ}, T^{rr}) and the pressure P′, from recovered primitives."""
@inline function fluxes(np::Float64, Tp::Float64, V::Float64)
    Γ  = 1/sqrt(max(1 - V*V, 1e-300))
    hh = h_fast(Tp)
    Pp = np*Tp
    Nr = np*Γ*V                       # N^r  = n′u′^r
    Trt = np*hh*Γ*Γ*V                 # T^{rτ} = (e′+P′)u′^r u′^τ
    Trr = np*hh*Γ*Γ*V*V + Pp          # T^{rr} = (e′+P′)(u′^r)² + P′
    return Nr, Trt, Trr, Pp
end

# ═══════════════════════ Drag / diffusion source ═════════════════════════════════════════════════
# In the FLUID rest frame the Fokker-Planck operator has the bath Jüttner as its fixed point, and the
# transfer to the charm sector is
#     S_mom^LRF = −η_D M ν*_r = −η_D M n w
#     S_ene^LRF =  η_D M n (T_fluid·B − A) ,   A = ⟨v²⟩ , B = ⟨(3M²+2p²)/E³⟩
# S_mom is EXACT (the κ/2 piece of ∫p_x C is a total derivative).  A and B are negative moments of E,
# so they are tabulated over (T′, w); the equilibrium identity T·B(T,0) = A(T,0) is gate G_EQ.
const _NTP, _NWS = 96, 96
const _TPT = exp.(range(log(0.01), log(2.0); length = _NTP))
const _WST = collect(range(0.0, 0.999; length = _NWS))
const _ATAB = zeros(_NTP, _NWS)
const _BTAB = zeros(_NTP, _NWS)
const _SRC_MASS = Ref(0.0)

function _gauss_legendre(n::Int)
    x = zeros(n); w = zeros(n)
    for i in 1:n
        z = cos(π*(i - 0.25)/(n + 0.5)); pp = 0.0
        for _ in 1:100
            p0, p1 = 1.0, 0.0
            for j in 1:n
                p2 = p1; p1 = p0
                p0 = ((2j - 1)*z*p1 - (j - 1)*p2)/j
            end
            pp = n*(z*p0 - p1)/(z^2 - 1); dz = p0/pp; z -= dz
            abs(dz) < 1e-15 && break
        end
        x[i] = z; w[i] = 2/((1 - z^2)*pp^2)
    end
    (x, w)
end
const _XQ, _WQ = _gauss_legendre(160)
const _PANELS = ((0.0, 2.0), (2.0, 12.0), (12.0, 400.0))

"""`(A, B) = (⟨v²⟩, ⟨(3M²+2p²)/E³⟩)` over the boosted Jüttner at `(Tp, w)`, in the fluid frame."""
function source_moments(Tp::Float64, w::Float64, M::Float64)
    γ = 1/sqrt(max(1 - w*w, 1e-300)); a = γ/Tp; b = γ*w/Tp
    ps = M*abs(b)/sqrt(max(a*a - b*b, 1e-300))
    emax = -a*sqrt(M*M + ps*ps) + abs(b)*ps
    n = A = B = 0.0
    for (plo, phi) in _PANELS
        hw, mid = 0.5*(phi - plo), 0.5*(phi + plo)
        for i in eachindex(_XQ)
            p = mid + hw*_XQ[i]; wp = hw*_WQ[i]*p*p
            E = sqrt(M*M + p*p)
            for j in eachindex(_XQ)
                g = wp*_WQ[j]*exp(-a*E + b*p*_XQ[j] - emax)
                n += g; A += g*(p/E)^2; B += g*(3*M*M + 2*p*p)/E^3
            end
        end
    end
    (A/n, B/n)
end

function build_source_tables!(M::Float64)
    _SRC_MASS[] == M && return nothing
    Threads.@threads :static for k in 1:_NTP
        for l in 1:_NWS
            _ATAB[k, l], _BTAB[k, l] = source_moments(_TPT[k], _WST[l], M)
        end
    end
    _SRC_MASS[] = M
    return nothing
end

@inline function _bilin(T, Tp, w)
    tt = clamp((log(clamp(Tp, _TPT[1], _TPT[end])) - log(_TPT[1]))/(log(_TPT[end]) - log(_TPT[1]))*(_NTP - 1) + 1, 1, _NTP)
    ww = clamp(abs(w)/_WST[end]*(_NWS - 1) + 1, 1, _NWS)
    k = clamp(floor(Int, tt), 1, _NTP - 1); l = clamp(floor(Int, ww), 1, _NWS - 1)
    a, b = tt - k, ww - l
    (1-a)*(1-b)*T[k,l] + a*(1-b)*T[k+1,l] + (1-a)*b*T[k,l+1] + a*b*T[k+1,l+1]
end

"""Drag rate η_D [fm⁻¹] from D_sT, in the production Langevin's convention: 1/η_D = M·D_sT/T²."""
@inline eta_drag(T::Float64, M::Float64, DsT::Float64) = T*T/(M*max(DsT, 1e-12))*fmGeV

"""Resolve `DsT` at temperature `T`.  A CALLABLE is accepted, exactly as `main2IS2.transport_all`
does (`DsT isa Function ? DsT(T) : DsT`), so LP1's `is2_dropin.jl` can hand either solver the same
`T -> lp1_effective_DsT(T; DsT_linear=true)` closure.  The env-flag form is kept for standalone runs."""
@inline function effective_DsT(T::Float64, DsT)
    DsT isa Function && return Float64(DsT(T))
    M1_DSTT_LINEAR || return Float64(DsT)
    M1_DSTT_SLOPE*max(T, M1_DSTT_TFO) + M1_DSTT_OFFSET
end

# ═══════════════════════ Right-hand side ═════════════════════════════════════════════════════════
@inline minmod(a, b) = (a*b <= 0) ? 0.0 : (abs(a) < abs(b) ? a : b)

"""Minmod-limited left/right face states of `u` at face `k` (between cells k and k+1).

⚠ SECOND ORDER IS NOT OPTIONAL.  A first-order Rusanov charm sector carries a numerical diffusivity
λΔr/2 which at Δr = 0.08 fm is comparable to the PHYSICAL D_s ≈ 0.15 fm — it silently replaces the
transport being solved for.  This is not hypothetical: it inverted the closure ranking once already
(`Tex/HeavyQuarkHydro/MAXENT_M1_PROGRAM.md` §4)."""
@inline function _faces(u, k, N)
    iL  = clamp(k, 1, N);     iR  = clamp(k + 1, 1, N)
    iLL = clamp(k - 1, 1, N); iRR = clamp(k + 2, 1, N)
    sL = minmod(u[iL] - u[iLL], u[iR] - u[iL])
    sR = minmod(u[iR] - u[iL],  u[iRR] - u[iR])
    (u[iL] + 0.5*sL, u[iR] - 0.5*sR)
end

"""One RHS evaluation.  `U = (qN, QE, QM)` are the Milne conserved densities τr(N^τ, T^{ττ}, T^{τr})."""
# TERM-BY-TERM DIAGNOSTIC for the r-momentum equation.  `diag` (when given) receives, per cell and in
# PHYSICAL units (divided by τr), the three contributions whose sum is ∂_τ T^{τr}:
#   diag[1] flux divergence −∂_r(τr T^{rr})/(τr)     diag[4] the same for the τ equation
#   diag[2] geometry        +P′/r                    diag[5] geometry, τ equation  (−P′/τ)
#   diag[3] drag source     S^r                      diag[6] drag source, τ equation  S^τ
#
# ⭐ THE PROJECTION IS WHAT MAKES THESE USABLE.  The raw terms are dominated by charm riding the bulk
# flow (T^{τr} ≈ n h u^τu^r), so they describe advection and say nothing about the diffusion current.
# Contracting with ū_ν = (u^r, −u^τ) — the unit spacelike vector orthogonal to u — removes it, and the
# SOURCE collapses exactly:
#     ū_ν S^ν = u^r(S_e u^τ + S_m u^r) − u^τ(S_e u^r + S_m u^τ) = −S_m = +η_D M n w ,
# the energy transfer projecting out identically since (u^τ)² − (u^r)² = 1.  So the projected equation
# is the local-rest-frame force balance on the current, and its only source is the drag.
# This exists because the freeze-out current is the residual of a ~10× cancellation between the first
# and the third, and a claim about which term is wrong must be read off the solver, not inferred from
# a quasi-static proxy built by differentiating its output splines.
function compute_dUdt!(dU, U, τ::Float64, grid::M1Grid1D, bg::M1Background;
                       M::Float64, DsT, T_floor::Float64, nbad::Ref{Int}, diag = nothing)
    Nr = length(grid.r); τs = max(τ, 1e-12)
    qN, QE, QM = U
    # primitives at cell centres (also returned to the caller through the scratch vectors)
    np = zeros(Nr); Tp = zeros(Nr); V = zeros(Nr)
    @inbounds for i in 1:Nr
        jac = τs*max(grid.r[i], 1e-12)
        n_, t_, v_, ok = recover(qN[i]/jac, QM[i]/jac, QE[i]/jac, M)
        np[i] = n_; Tp[i] = t_; V[i] = v_; ok || (nbad[] += 1)
    end
    for c in 1:3; fill!(dU[c], 0.0); end

    # ── fluxes: Rusanov on the PHYSICAL densities, the whole numerical flux then area-weighted ──
    # 🔴🔴 RECONSTRUCT AND DISSIPATE THE PHYSICAL DENSITIES, NEVER THE τr-WEIGHTED ONES.  Doing the
    # latter looks natural (they are what is conserved) and is catastrophically wrong at the origin:
    # τr varies by a factor 3 between the first two cells, so the Rusanov jump ½λ(U_R − U_L) is
    # proportional to the VALUE rather than to a physical gradient and pumps charm through r = 0.
    # Measured before the fix: n(r→0) grew 72× over τ = 2→10 fm/c, T′ dropped to 0.85 T at the axis,
    # and the resulting inward pressure gradient made the whole blob CONTRACT (⟨r²⟩ 50 → 30 fm²) —
    # a negative diffusion coefficient.  The correct form is
    #     F_face = τ r_F · [ ½(F_L + F_R) − ½λ (U^phys_R − U^phys_L) ] ,
    # which vanishes identically at r = 0 because the area does.
    # λ = 1 is the correct wave-speed bound: every flux is a moment of a quantity bounded by c, so no
    # characteristic can exceed it.  Being generous costs O(Δr²) once the reconstruction is second
    # order; being exact would cost a 3×3 Jacobian per face.
    Dc = zeros(Nr); Ec = zeros(Nr); Mc = zeros(Nr)
    @inbounds for i in 1:Nr
        jac = τs*max(grid.r[i], 1e-12)
        Dc[i] = qN[i]/jac; Ec[i] = QE[i]/jac; Mc[i] = QM[i]/jac
    end
    Fp1 = Fp2 = Fp3 = 0.0
    @inbounds for k in 0:Nr
        jacf = τs*max(grid.rF[k + 1], 0.0)
        aL, aR = _faces(Dc, k, Nr); bL, bR = _faces(Ec, k, Nr); cL, cR = _faces(Mc, k, Nr)
        nL, tL, vL, _ = recover(aL, cL, bL, M)
        nR, tR, vR, _ = recover(aR, cR, bR, M)
        NrL, TrtL, TrrL, _ = fluxes(nL, tL, vL)
        NrR, TrtR, TrrR, _ = fluxes(nR, tR, vR)
        f1 = jacf*(0.5*(NrL  + NrR)  - 0.5*(aR - aL))
        f2 = jacf*(0.5*(TrtL + TrtR) - 0.5*(bR - bL))
        f3 = jacf*(0.5*(TrrL + TrrR) - 0.5*(cR - cL))
        if k >= 1
            dU[1][k] -= (f1 - Fp1)/grid.dr[k]
            dU[2][k] -= (f2 - Fp2)/grid.dr[k]
            dU[3][k] -= (f3 - Fp3)/grid.dr[k]
            if diag !== nothing
                jk = τs*max(grid.r[k], 1e-12)
                diag[1][k] = -(f3 - Fp3)/grid.dr[k]/jk
                diag[4][k] = -(f2 - Fp2)/grid.dr[k]/jk
            end
        end
        Fp1, Fp2, Fp3 = f1, f2, f3
    end

    # ── geometry + drag/diffusion sources ───────────────────────────────────────────────────────
    @inbounds for i in 1:Nr
        r = grid.r[i]
        Pp = np[i]*Tp[i]
        dU[2][i] -= r*Pp                      # Γ^τ_{ηη}T^{ηη} = P′/τ  ⇒  −τr·P′/τ
        diag === nothing || (diag[5][i] = -r*Pp/(τs*max(r, 1e-12)))
        dU[3][i] += τs*Pp                     # Γ^r_{φφ}T^{φφ} = −P′/r ⇒  +τr·P′/r
        diag === nothing || (diag[2][i] = τs*Pp/(τs*max(r, 1e-12)))

        np[i] > M1_N_FLOOR || continue
        T = max(bg_T(bg, τ, r), T_floor)
        uτ, ur = _u_from_ur(bg_ur(bg, τ, r)); v = ur/uτ
        # the boost RELATIVE TO THE FLUID: relativistic velocity subtraction of the background flow
        w = clamp((V[i] - v)/(1 - V[i]*v), -0.999999, 0.999999)
        γw = 1/sqrt(1 - w*w)
        n_fluid = np[i]*γw                    # u·N = n′γ_w
        ηD = eta_drag(T, M, effective_DsT(T, DsT))
        A = _bilin(_ATAB, Tp[i], w); B = _bilin(_BTAB, Tp[i], w)
        Smom = -ηD*M*n_fluid*w                # exact
        Sene =  ηD*M*n_fluid*(T*B - A)
        # LRF → Milne:  S^μ = S_ene u^μ + S_mom ū^μ ,  ū^μ = (u^r, u^τ)
        jac = τs*max(r, 1e-12)
        dU[2][i] += jac*(Sene*uτ + Smom*ur)
        dU[3][i] += jac*(Sene*ur + Smom*uτ)
        if diag !== nothing
            diag[3][i] = Sene*ur + Smom*uτ
            diag[6][i] = Sene*uτ + Smom*ur
        end
    end
    return np, Tp, V
end

# ═══════════════════════ Driver ══════════════════════════════════════════════════════════════════
"""
    solve_M1(grid, α0, νr0, τ0, τf, bg; ...) -> Dict

SSP-RK2 in τ.  Returns the same keys main2IS2's `run_static_IS2_test` does — `tau`, `r`, `n`, `nur`,
`alpha` — plus the M1-specific `Tprime`, `w` and `eps`, and a `diagnostics` entry whose
`realizability_floor` counts cell-evaluations that fell below the cold-beam floor.  A nonzero count
is a finding, not a nuisance: it means the discretisation, not the formulation, left the cone.
"""
function solve_M1(grid::M1Grid1D, α0::Vector{Float64}, νr0::Vector{Float64},
                  τ0::Float64, τf::Float64, bg::M1Background;
                  CFL::Float64 = 0.25, save_dt::Float64 = 0.1, DsT = 0.1163,
                  T_floor::Float64 = 1e-6, eos = LatticeHRGEOS(canon_factor = 1.0),
                  log_every::Int = 100, Tprime0 = nothing)
    Nr = length(grid.r); M = hq_mass(eos)
    _build_h_table!(M); build_source_tables!(M)

    # ── initial conserved state from (α, ν^r) on the background flow ────────────────────────────
    # The IC is the boosted Jüttner carrying exactly the supplied (n, ν^r) at the FLUID temperature:
    # w = ν*_r/n = ν^r/(n u^τ), T′ = T.  That is the same state IS2 starts from, so the two solvers
    # begin from an identical charm configuration and any difference is dynamics.
    qN = zeros(Nr); QE = zeros(Nr); QM = zeros(Nr)
    for i in 1:Nr
        r = grid.r[i]; T = max(bg_T(bg, τ0, r), T_floor)
        _, n_eq, _ = eos_Pne(T, α0[i]*T, eos)
        n = max(n_eq, 0.0)
        uτ, ur = _u_from_ur(bg_ur(bg, τ0, r))
        w = n > M1_N_FLOOR ? clamp(νr0[i]/(n*uτ), -0.999999, 0.999999) : 0.0
        γw = 1/sqrt(1 - w*w)
        upτ = γw*(uτ + w*ur); upr = γw*(ur + w*uτ)      # eq:boostsubs
        np = n/γw                                        # n′ so that u·N = n exactly
        # `Tprime0` exists so the ENERGY RELAXATION is testable.  Every gate that starts at T′ = T
        # exercises the energy equation only through the geometry terms, and a source that failed to
        # reheat the charm toward the bath would pass all of them — which is exactly what happened.
        Tp0 = Tprime0 === nothing ? T : Float64(Tprime0 isa Function ? Tprime0(r) : Tprime0)
        hh = np*hpp(Tp0, M)
        jac = τ0*max(r, 1e-12)
        qN[i] = jac*np*upτ
        QE[i] = jac*(hh*upτ*upτ - np*Tp0)
        QM[i] = jac*(hh*upτ*upr)
    end

    U  = (qN, QE, QM)
    U1 = (similar(qN), similar(QE), similar(QM))
    dU = (similar(qN), similar(QE), similar(QM))
    nbad = Ref(0)
    dg = M1_DIAG_TERMS ? ntuple(_ -> zeros(Nr), 8) : nothing
    dgsnaps = Vector{NTuple{8,Vector{Float64}}}()
    τ = τ0; step = 0
    dr_min = minimum(grid.dr)
    snaps_τ = Float64[]; snaps = Vector{NTuple{6,Vector{Float64}}}()
    next_save = τ0

    while τ < τf - 1e-12
        dt = min(CFL*dr_min, τf - τ)                      # λ ≤ 1, so this is the CFL
        compute_dUdt!(dU, U, τ, grid, bg; M = M, DsT = DsT, T_floor = T_floor, nbad = nbad, diag = dg)
        # slots 7,8 hold ∂_τ Q_E and ∂_τ Q_M per unit τr — the left-hand sides, so the projected
        # balance can be checked against its own time derivative rather than assumed quasi-static.
        if dg !== nothing
            for i in 1:Nr
                jj = max(τ, 1e-12)*max(grid.r[i], 1e-12)
                dg[7][i] = dU[2][i]/jj; dg[8][i] = dU[3][i]/jj
            end
        end
        for c in 1:3; @. U1[c] = U[c] + dt*dU[c]; end
        compute_dUdt!(dU, U1, τ + dt, grid, bg; M = M, DsT = DsT, T_floor = T_floor, nbad = nbad)
        for c in 1:3; @. U[c] = 0.5*(U[c] + U1[c] + dt*dU[c]); end
        τ += dt; step += 1

        if τ >= next_save - 1e-12 || τ >= τf - 1e-12
            push!(snaps_τ, τ); push!(snaps, _snapshot(U, τ, grid, bg, M, T_floor, eos))
            dg === nothing || push!(dgsnaps, ntuple(c -> copy(dg[c]), 8))
            next_save += save_dt
        end
        log_every > 0 && step % log_every == 0 &&
            @info @sprintf("M1 τ=%.3f dt=%.4f steps=%d floored=%d", τ, dt, step, nbad[])
    end

    # ⚠ THE OUTPUT CONTRACT IS main2IS2's, EXACTLY: `r_grid`/`t_grid` and (Nr, Nt) arrays.  A
    # drop-in replacement that returns its own key names and its own array orientation is not a
    # drop-in — and a transposed field read through the wrong convention is silent, not an error.
    Nt = length(snaps_τ)
    grab(k) = (A = zeros(Nr, Nt); for j in 1:Nt, i in 1:Nr; A[i, j] = snaps[j][k][i]; end; A)
    return Dict{String,Any}(
        "r_grid" => copy(grid.r), "t_grid" => collect(snaps_τ),
        "n" => grab(1), "nur" => grab(2), "alpha" => grab(3),
        # M1-specific, no IS2 counterpart: the charm's own temperature, its boost relative to the
        # fluid, and N^τ (the density that is actually conserved, as opposed to the fluid-frame n).
        "Tprime" => grab(4), "w" => grab(5), "Ntau" => grab(6),
        "diagnostics" => Dict("realizability_floor" => nbad[], "steps" => step),
        # (Nr, Nt) like everything else; only present when FIVO_M1_DIAG_TERMS=1
        "mom_flux" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 1, Nr) : nothing,
        "mom_geom" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 2, Nr) : nothing,
        "mom_drag" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 3, Nr) : nothing,
        "ene_flux" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 4, Nr) : nothing,
        "ene_geom" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 5, Nr) : nothing,
        "ene_drag" => M1_DIAG_TERMS ? _grabdg(dgsnaps, 6, Nr) : nothing,
        "ene_ddt"  => M1_DIAG_TERMS ? _grabdg(dgsnaps, 7, Nr) : nothing,
        "mom_ddt"  => M1_DIAG_TERMS ? _grabdg(dgsnaps, 8, Nr) : nothing,
    )
end

"""Fluid-frame observables from the conserved state: (n, ν^r, α, T′, w)."""
function _snapshot(U, τ, grid::M1Grid1D, bg::M1Background, M, T_floor, eos)
    Nr = length(grid.r); τs = max(τ, 1e-12)
    n_a = zeros(Nr); ν_a = zeros(Nr); α_a = zeros(Nr); T_a = zeros(Nr); w_a = zeros(Nr)
    Nt_a = zeros(Nr)                                   # N^τ: the density that is CONSERVED
    qN, QE, QM = U
    @inbounds for i in 1:Nr
        r = grid.r[i]; jac = τs*max(r, 1e-12)
        np, Tp, V, _ = recover(qN[i]/jac, QM[i]/jac, QE[i]/jac, M)
        T = max(bg_T(bg, τ, r), T_floor)
        uτ, ur = _u_from_ur(bg_ur(bg, τ, r)); v = ur/uτ
        w = np > M1_N_FLOOR ? clamp((V - v)/(1 - V*v), -0.999999, 0.999999) : 0.0
        γw = 1/sqrt(1 - w*w)
        n = np*γw                                   # u·N
        _, n_eq, _ = eos_Pne(T, 0.0, eos)
        n_a[i] = n
        ν_a[i] = n*w*uτ                             # ν^r = n w u^τ  (LRF component is n w)
        α_a[i] = n_eq > TINY && n > M1_N_FLOOR ? log(n/n_eq) : -700.0
        T_a[i] = Tp; w_a[i] = w
        Nt_a[i] = qN[i]/jac
    end
    (n_a, ν_a, α_a, T_a, w_a, Nt_a)
end

function run_static_M1_test(; background_file::String, DsT = 0.1163,
                            τ0::Float64 = 0.4, τfinal::Float64 = 8.0,
                            Nr::Int = 300, rmax::Float64 = 25.0, CFL::Float64 = 0.25,
                            dump_dt::Float64 = 0.1, T_floor::Float64 = 1e-6,
                            init_mode::Symbol = :auto,
                            n_profile = r -> exp(-r^2/(2.0*4.0^2)), nur_profile = nothing,
                            eos = LatticeHRGEOS(canon_factor = 1.0), log_every::Int = 100)
    bg = load_M1_background(background_file)
    grid = M1Grid1D(Nr, min(rmax, last(bg.r_grid)))
    use_bg = init_mode === :background || (init_mode === :auto && bg.α_spl !== nothing)

    α = zeros(Nr); νr = zeros(Nr)
    for i in 1:Nr
        r = grid.r[i]; T = max(bg_T(bg, τ0, r), T_floor)
        if use_bg
            α[i] = bg_al(bg, τ0, r); νr[i] = bg_nur(bg, τ0, r)
        else
            _, n_eq0, _ = eos_Pne(T, 0.0, eos)
            α[i] = n_eq0 > TINY ? log(max(n_profile(r)/n_eq0, TINY)) : 0.0
            νr[i] = nur_profile === nothing ? 0.0 : Float64(nur_profile(r))
        end
    end
    solve_M1(grid, α, νr, τ0, τfinal, bg; CFL = CFL, save_dt = dump_dt, DsT = DsT,
             T_floor = T_floor, eos = eos, log_every = log_every)
end

end # module hydro_current_M1
