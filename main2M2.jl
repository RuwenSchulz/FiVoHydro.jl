module hydro_current_M2
# ═════════════════════════════════════════════════════════════════════════════════════════════════
# main2M2.jl — THE FOUR-FIELD (M2) MAXIMUM-ENTROPY CHARM SYSTEM.
#
# A PARALLEL module to main2M1.jl, never a modification of it (the CMExperiment pattern): M1 and
# production stay byte-identical and all solvers run on the same background.
#
# ── THE SYSTEM (derived + machine-verified in Julia/tools/m2_milne_balance.jl, 9/9) ─────────────
# The carrier is the MILNE-STATIC-frame maximum-entropy state on FOUR moments
#
#     f = exp(−a − b p^τ̂ + c p^r̂ + e p^r̂/p^τ̂)
#
# (frame decision 2026-08-20: the fluid-tied variant has a FOLD in its conservative recovery at
# v ≈ 0.7–0.9 — m2_cone_scan.jl G6/G7 — so the family is tied to the computational frame, where
# the inversion is globally injective: derive_inversion_wellposed_m2.wl, 16/16).  Conserved
# densities q = τr·(N^τ, T^{ττ}, T^{τr}, N^r):
#
#     ∂_τ q_N + ∂_r(τr N^r)    = 0
#     ∂_τ Q_E + ∂_r(τr T^{τr}) = τr S_E − r P⊥
#     ∂_τ Q_M + ∂_r(τr T^{rr}) = τr S_M + τ P⊥
#     ∂_τ q_ν + ∂_r(τr K^{rr}) = τr S_ν + τ K^{φφ} + r K^{rηη}
#
# with P⊥ = <pφ̂²/p^τ̂> (the transverse pressure, ANISOTROPIC for e ≠ 0; = n′T′ at e = 0),
# K^{ab} = <p^â p^b̂/(p^τ̂)²>-class moments.  ν̂ = N^r IS the number flux and T^{τr} IS the energy
# flux (moment-hierarchy overlap) — only (T^{rr}, P⊥, K^{rr}, K^{φφ}, K^{rηη}) carry the closure.
#
# ── CLOSURE EVALUATION: LIVE 1-D QUADRATURE, NO TABLES ─────────────────────────────────────────
# The cosθ integrals are analytic (S₀..S₃ sinh forms), so every closure moment is ONE p-panel
# quadrature (~10⁻⁵ s).  The primitive inversion is the staged scheme gated in m2_cone_scan.jl
# (5/5): M1 scalar-bisection seed at e = 0 → 3-D Newton with the analytic covariance Jacobian →
# guaranteed monotone 1-D fallback (Illinois on e along the (Y,X) sheet).  Fallback activations,
# cone floors and e-clamps are COUNTED and reported, never hidden (the M1 contract).
#
# ── HYPERBOLICITY / CFL ────────────────────────────────────────────────────────────────────────
# The system is strictly hyperbolic with max|λ| = 0.984 < 1 over the state lattice
# (m2_hyperbolicity.jl, 6/6) ⇒ Rusanov λ = 1 and the M1 CFL carry over unchanged.
#
# ── SOURCES: THE PRODUCTION DRIFT ṗ = −η_D·M·p/E ───────────────────────────────────────────────
# (velocity-proportional, LangevInMedium tau_drag convention — what makes S_mom = −η_D M n w
# exact and τ_current = h/(η_D M); the by-parts forms below are gated in m2_milne_balance S1w-S3w).
# In the fluid frame at local velocity v, with κ = 2MTη_D and γ = 1/√(1−v²):
#     S_E = −η_D M γ <p²/E² + v p_r/E>  +  (κ/2) γ <(2E²+M²)/E³>
#     S_M = −η_D M γ <p_r/E + v p²/E²>  +  (κ/2) γ v <(2E²+M²)/E³>
#     S_ν = −η_D M <(p/E)·∇w̃>          +  (κ/2) <∇²w̃> ,   w̃ = (p_r + vE)/(E + v p_r)
# evaluated as one 2-D fluid-frame quadrature per cell (threaded).
#
# Entry points mirror main2IS2/main2M1 so the LP1 drop-in can call any of the three:
#     run_static_M2_test(; background_file, DsT, τ0, τfinal, Nr, rmax, ...) -> Dict
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

export run_static_M2_test, solve_M2, M2Grid1D, M2Background, load_M2_background

# ── knobs (all diagnostic; none is a regulator) ─────────────────────────────────────────────────
const M2_TP_LO   = parse(Float64, get(ENV, "FIVO_M2_TP_LO", "1e-5"))
const M2_TP_HI   = parse(Float64, get(ENV, "FIVO_M2_TP_HI", "5.0"))
const M2_N_FLOOR = parse(Float64, get(ENV, "FIVO_M2_N_FLOOR", "1e-300"))
const M2_E_MAX   = parse(Float64, get(ENV, "FIVO_M2_E_MAX", "25.0"))   # |e| clamp bound, counted
const M2_NEWTON_TOL = parse(Float64, get(ENV, "FIVO_M2_NEWTON_TOL", "1e-10"))

# ═══════════════════════ Grid / background (same shapes as main2IS2/M1) ══════════════════════════
struct M2Grid1D
    r::Vector{Float64}
    rF::Vector{Float64}
    dr::Vector{Float64}
end
function M2Grid1D(Nr::Int, rmax::Float64)
    d = rmax/Nr
    M2Grid1D([(i - 0.5)*d for i in 1:Nr], [(i - 1)*d for i in 1:(Nr + 1)], fill(d, Nr))
end

struct M2Background
    r_grid::Vector{Float64}
    t_grid::Vector{Float64}
    T_spl::Any
    ur_spl::Any
    α_spl::Any
    n_spl::Any
    nur_spl::Any
end
function load_M2_background(path::AbstractString)
    isfile(path) || error("Background file not found: $path")
    jldopen(path, "r") do f
        _k(s) = haskey(f, s) ? s : haskey(f, s*"1") ? s*"1" : s
        _o(s) = (haskey(f, s) || haskey(f, s*"1")) ? f[_k(s)] : nothing
        M2Background(Float64.(f["r_grid"]), Float64.(f["t_grid"]), f[_k("T_spline")],
                     _o("ur_spline"), _o("α_spline"), _o("n_spline"), _o("nur_spline"))
    end
end
@inline function _clamp_eval(spl, r, τ, rg, tg)
    spl === nothing && return 0.0
    Float64(spl(clamp(Float64(r), first(rg), last(rg)), clamp(Float64(τ), first(tg), last(tg))))
end
@inline bg_T(bg::M2Background, τ, r)   = _clamp_eval(bg.T_spl,   r, τ, bg.r_grid, bg.t_grid)
@inline bg_ur(bg::M2Background, τ, r)  = _clamp_eval(bg.ur_spl,  r, τ, bg.r_grid, bg.t_grid)
@inline bg_al(bg::M2Background, τ, r)  = _clamp_eval(bg.α_spl,   r, τ, bg.r_grid, bg.t_grid)
@inline bg_nur(bg::M2Background, τ, r) = _clamp_eval(bg.nur_spl, r, τ, bg.r_grid, bg.t_grid)
@inline _u_from_ur(ur) = (sqrt(1.0 + Float64(ur)^2), Float64(ur))

# ═══════════════════════ Jüttner scalars + h-table (M1's, for the e = 0 seed) ════════════════════
@inline function _kx(z)
    zc = clamp(Float64(z), 1e-8, 700.0)
    (besselkx(1, zc), besselkx(2, zc))
end
@inline function ebar(Tp, M)
    Tc = max(Float64(Tp), M2_TP_LO)
    K1x, K2x = _kx(M/Tc)
    3*Tc + M*K1x/max(K2x, 1e-300)
end
@inline hpp(Tp, M) = ebar(Tp, M) + max(Float64(Tp), M2_TP_LO)

const _NH = 8000
const _HTAB = Ref{Vector{Float64}}(Float64[])
const _HSC  = Ref(0.0)
const _HMASS = Ref(0.0)
function _build_h_table!(M)
    (_HMASS[] == M && !isempty(_HTAB[])) && return nothing
    _HTAB[] = [hpp(T, M) for T in exp.(range(log(M2_TP_LO), log(M2_TP_HI); length = _NH))]
    _HSC[]  = (_NH - 1)/(log(M2_TP_HI) - log(M2_TP_LO))
    _HMASS[] = M
    return nothing
end
@inline function h_fast(Tp)
    t = (log(clamp(Tp, M2_TP_LO, M2_TP_HI)) - log(M2_TP_LO))*_HSC[] + 1
    k = clamp(floor(Int, t), 1, _NH - 1); θ = t - k
    H = _HTAB[]
    (1 - θ)*H[k] + θ*H[k + 1]
end

# ═══════════════════════ The closure: live 1-D quadrature over the family ════════════════════════
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
const _XQ, _WQ = _gauss_legendre(128)
const _XS, _WS = _gauss_legendre(64)                       # the 2-D source quadrature

"""Angular integrals ∫μ^k e^{Aμ}dμ, k = 0..3, PRE-multiplied by e^{mbE} (mbE = −b·E ≤ 0), in
overflow/cancellation-safe form (small |A| by series)."""
@inline function _sk(A::Float64, mbE::Float64)
    if abs(A) < 1e-4
        ex = exp(mbE)
        s0 = ex*(2 + A*A/3)
        s1 = ex*(2A/3)*(1 + A*A/10)
        s2 = ex*(2/3 + A*A/5)
        s3 = ex*(2A/5)*(1 + 5A*A/42)
        return s0, s1, s2, s3
    end
    ep = exp(mbE + A); em = exp(mbE - A)
    s0 = (ep - em)/A
    s1 = (ep*(A - 1) + em*(A + 1))/A^2
    s2 = (ep*(A*A - 2A + 2) - em*(A*A + 2A + 2))/A^3
    s3 = (ep*(A^3 - 3A*A + 6A - 6) + em*(A^3 + 3A*A + 6A + 6))/A^4
    s0, s1, s2, s3
end

"""
    closure_moments(b, c, e, M) -> NamedTuple

All raw (a = 0) moments of the family the solver needs, from ONE p-panel pass:
state m1=<1>, mE=<P>, mp=<pr>, mn=<pr/P>; fluxes/geometry mTrr=<pr²/P>, mKrr=<pr²/P²>,
mPp=<pφ²/P>, mKph=<pφ²/P²>, mKre=<pr pη²/P³>; covariance second moments mEE, mEp, mpp.
Realizability b > |c| is the caller's contract (structural after recovery).
"""
function closure_moments(b::Float64, c::Float64, e::Float64, M::Float64)
    pmax = max(80.0, 45.0/max(b - abs(c), 1e-3))
    m1 = mE = mp = mn = mTrr = mKrr = mPp = mKph = mKre = mEE = mEp = mpp = 0.0
    plo = 0.0
    for pedge in (2.0, 12.0, 60.0, 240.0, 960.0, 3840.0, 15360.0)
        phi = min(pedge, pmax)
        if phi > plo
            hw, mid = 0.5*(phi - plo), 0.5*(phi + plo)
            @inbounds for i in eachindex(_XQ)
                p = mid + hw*_XQ[i]; wq = hw*_WQ[i]*p*p
                P = sqrt(M*M + p*p)
                s0, s1, s2, s3 = _sk((c + e/P)*p, -b*P)
                m1   += wq*s0
                mE   += wq*P*s0
                mp   += wq*p*s1
                mn   += wq*(p/P)*s1
                mTrr += wq*(p*p/P)*s2
                mKrr += wq*(p*p/(P*P))*s2
                mPp  += wq*(p*p/(2P))*(s0 - s2)
                mKph += wq*(p*p/(2P*P))*(s0 - s2)
                mKre += wq*(p*p*p/(2P^3))*(s1 - s3)
                mEE  += wq*P*P*s0
                mEp  += wq*p*P*s1
                mpp  += wq*p*p*s2
            end
        end
        plo = pedge
        plo >= pmax && break
    end
    (; m1, mE, mp, mn, mTrr, mKrr, mPp, mKph, mKre, mEE, mEp, mpp)
end

"ratios (Y, X, N) and their analytic covariance Jacobian over (b, c, e), from one moment pass"
function _ratios_jac(b, c, e, M)
    m = closure_moments(b, c, e, M)
    D = m.m1
    Y, X, N = m.mE/D, m.mp/D, m.mn/D
    cEE = m.mEE/D - Y*Y;  cEp = m.mEp/D - Y*X;  cEn = m.mp/D - Y*N     # <P·pr/P> = <pr>
    cpp = m.mpp/D - X*X;  cpn = m.mTrr/D - X*N; cnn = m.mKrr/D - N*N
    J = [-cEE  cEp  cEn; -cEp  cpp  cpn; -cEn  cpn  cnn]
    ((Y, X, N), J, m)
end

# ═══════════════════════ Primitive recovery (staged; m2_cone_scan.jl scheme) ═════════════════════
"""
    recover_m2(D, E, S, ν, M; b0, c0, e0) -> (b, c, e, mom, ok, flags)

`D = N^τ`, `E = T^{ττ}`, `S = T^{τr}`, `ν = N^r` at a cell/face.  Staged inversion:
M1 scalar bisection at e = 0 (or the supplied warm start) → damped 3-D Newton with the analytic
covariance Jacobian → Illinois on e along the (Y,X) sheet (guaranteed: N(e) monotone).
`flags`: 0 ok · 1 cone floor (E/D below the cold beam; placed on it, e = 0) · 2 e clamped at
±M2_E_MAX · 3 fallback used (converged) · 4 failed (kept at best iterate).  All counted upstream.
"""
function recover_m2(D::Float64, E::Float64, S::Float64, ν::Float64, M::Float64;
                    b0::Float64 = NaN, c0::Float64 = NaN, e0::Float64 = 0.0)
    D > M2_N_FLOOR || return (1.0/M2_TP_LO, 0.0, 0.0, nothing, true, 0)
    Y = E/D; X = S/D; N = ν/D
    scale = max(1.0, abs(Y), abs(X))
    # cone floor (the M1 cold-beam bound; e adds no room below it — A2 CHECK 4)
    if Y < sqrt(M*M + X*X)
        h = h_fast(M2_TP_LO); V = X/sqrt(h*h + X*X)
        γ = 1/sqrt(max(1 - V*V, 1e-300)); Tp = M2_TP_LO
        return (γ/Tp, γ*V/Tp, 0.0, nothing, false, 1)
    end
    # seed: warm start if supplied, else M1 bisection at e = 0
    b, c, e = b0, c0, e0
    if !isfinite(b) || b <= abs(c)
        G(Tp) = (h = h_fast(Tp); h*h + X*X - Tp*h - Y*sqrt(h*h + X*X))
        lo, hi = M2_TP_LO, M2_TP_HI
        @inbounds for _ in 1:45
            mid = sqrt(lo*hi)
            G(mid) < 0 ? (lo = mid) : (hi = mid)
        end
        Tp = sqrt(lo*hi); h = h_fast(Tp); V = X/sqrt(h*h + X*X)
        γ = 1/sqrt(max(1 - V*V, 1e-300))
        b = γ/Tp; c = γ*V/Tp; e = 0.0
    end
    # 3-D Newton with backtracking
    local mom
    for it in 1:30
        (R, J, m) = _ratios_jac(b, c, e, M)
        mom = m
        r1 = R[1] - Y; r2 = R[2] - X; r3 = R[3] - N
        rn0 = max(abs(r1), abs(r2), abs(r3))
        rn0 < M2_NEWTON_TOL*scale && return (b, c, e, m, true, 0)
        d = try
            -(J \ [r1, r2, r3])
        catch
            break
        end
        all(isfinite, d) || break
        s = 1.0
        while b + s*d[1] <= abs(c + s*d[2])*(1 + 1e-12) + 1e-10 || b + s*d[1] <= 0
            s *= 0.5; s < 1e-8 && break
        end
        moved = false
        while s >= 1e-8
            (Rn, _, _) = _ratios_jac(b + s*d[1], c + s*d[2], e + s*d[3], M)
            if all(isfinite, Rn) && max(abs(Rn[1] - Y), abs(Rn[2] - X), abs(Rn[3] - N)) < rn0
                b += s*d[1]; c += s*d[2]; e += s*d[3]; moved = true
                break
            end
            s *= 0.5
        end
        moved || break
        abs(e) > M2_E_MAX && ((b, c, ok2) = _sheet_bc(Y, X, sign(e)*M2_E_MAX, M; b0 = b, c0 = c);
                              return (b, c, sign(e)*M2_E_MAX, closure_moments(b, c, sign(e)*M2_E_MAX, M),
                                      ok2, 2))
    end
    # fallback: Illinois on e along the (Y, X) sheet (m2_cone_scan G4: N(e) strictly monotone)
    b2, c2, e2, ok2 = _fallback_e(Y, X, N, M)
    if ok2
        return (b2, c2, e2, closure_moments(b2, c2, e2, M), true, 3)
    end
    return (b, c, e, mom, false, 4)
end

"2-D damped Newton holding (Y, X) at fixed e — the sheet solve of the 1-D reduction"
function _sheet_bc(Y, X, e, M; b0 = 2.0, c0 = 0.0)
    b, c = b0, c0
    scale = max(1.0, abs(Y), abs(X))
    for it in 1:60
        (R, J, _) = _ratios_jac(b, c, e, M)
        r1 = R[1] - Y; r2 = R[2] - X
        rn0 = max(abs(r1), abs(r2))
        rn0 < 1e-11*scale && return (b, c, true)
        d = try
            -([J[1,1] J[1,2]; J[2,1] J[2,2]] \ [r1, r2])
        catch
            return (b, c, false)
        end
        s = 1.0
        while b + s*d[1] <= abs(c + s*d[2])*(1 + 1e-12) + 1e-10 || b + s*d[1] <= 0
            s *= 0.5; s < 1e-8 && return (b, c, rn0 < 1e-7*scale)
        end
        while s >= 1e-8
            (Rn, _, _) = _ratios_jac(b + s*d[1], c + s*d[2], e, M)
            if all(isfinite, Rn) && max(abs(Rn[1] - Y), abs(Rn[2] - X)) < rn0
                break
            end
            s *= 0.5
        end
        s < 1e-8 && return (b, c, rn0 < 1e-7*scale)
        b += s*d[1]; c += s*d[2]
    end
    (b, c, false)
end

function _fallback_e(Y, X, N, M)
    # seed on the sheet at e = 0 from the M1 bisection
    G(Tp) = (h = h_fast(Tp); h*h + X*X - Tp*h - Y*sqrt(h*h + X*X))
    lo, hi = M2_TP_LO, M2_TP_HI
    for _ in 1:45
        mid = sqrt(lo*hi)
        G(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    Tp = sqrt(lo*hi); h = h_fast(Tp); V = X/sqrt(h*h + X*X)
    γ = 1/sqrt(max(1 - V*V, 1e-300))
    b, c, ok = _sheet_bc(Y, X, 0.0, M; b0 = γ/Tp, c0 = γ*V/Tp)
    ok || return (b, c, 0.0, false)
    Nof(ee, bb, cc) = ((b2, c2, ok2) = _sheet_bc(Y, X, ee, M; b0 = bb, c0 = cc);
                       ok2 ? ((m = closure_moments(b2, c2, ee, M); (m.mn/m.m1, b2, c2, true))) :
                             (NaN, bb, cc, false))
    m0 = closure_moments(b, c, 0.0, M); N0 = m0.mn/m0.m1
    abs(N0 - N) < 1e-10 && return (b, c, 0.0, true)
    dir = N > N0 ? 1.0 : -1.0
    step = 0.5; efar = 0.0; Nfar = N0; bfar, cfar = b, c
    while (dir > 0 ? Nfar < N : Nfar > N)
        (Nt, bt, ct, ok2) = Nof(efar + dir*step, bfar, cfar)
        if !ok2
            step *= 0.5
            step < 1e-3 && return (bfar, cfar, efar, false)
            continue
        end
        efar += dir*step; Nfar = Nt; bfar, cfar = bt, ct
        step = min(step*2, 8.0)
        abs(efar) > M2_E_MAX && return (bfar, cfar, sign(efar)*M2_E_MAX, false)
    end
    e_lo, e_hi = dir > 0 ? (0.0, efar) : (efar, 0.0)
    flo = (dir > 0 ? N0 : Nfar) - N; fhi = (dir > 0 ? Nfar : N0) - N
    bb, cc = bfar, cfar; side = 0
    for _ in 1:120
        e_m = (fhi != flo) ? e_lo - flo*(e_hi - e_lo)/(fhi - flo) : 0.5*(e_lo + e_hi)
        (e_m <= e_lo || e_m >= e_hi) && (e_m = 0.5*(e_lo + e_hi))
        (Nm, bb, cc, ok2) = Nof(e_m, bb, cc)
        ok2 || return (bb, cc, e_m, false)
        fm = Nm - N
        abs(fm) < 1e-10 && return (bb, cc, e_m, true)
        if fm < 0
            e_lo, flo = e_m, fm
            side == -1 && (fhi *= 0.5); side = -1
        else
            e_hi, fhi = e_m, fm
            side == 1 && (flo *= 0.5); side = 1
        end
        e_hi - e_lo < 1e-14*max(1.0, abs(e_lo)) && return (bb, cc, e_m, abs(fm) < 1e-8)
    end
    (bb, cc, 0.5*(e_lo + e_hi), false)
end

# ═══════════════════════ Drag / diffusion sources (production drift ṗ = −η_D M p/E) ══════════════
@inline eta_drag(T::Float64, M::Float64, DsT::Float64) = T*T/(M*max(DsT, 1e-12))*fmGeV
@inline function effective_DsT(T::Float64, DsT)
    DsT isa Function && return Float64(DsT(T))
    Float64(DsT)
end

"""
    sources_m2(b, c, e, scale, v, T, ηD, M) -> (S_E, S_M, S_ν)

One 2-D fluid-frame quadrature (pr*, q*) per cell.  The family in fluid variables carries the
exact boost (p^τ̂ = γ(E + v pr*), p^r̂ = γ(pr* + vE)); the by-parts integrands are the CLOSED
forms gated in m2_milne_balance.jl S1w–S3w with κ = 2MTη_D.  `scale` = D/m1 fixes the physical
normalization (a is frame-independent).
"""
function sources_m2(b::Float64, c::Float64, e::Float64, scale::Float64,
                    v::Float64, T::Float64, ηD::Float64, M::Float64)
    γ = 1/sqrt(max(1 - v*v, 1e-300))
    κ2 = M*T*ηD                                          # κ/2
    bfl = γ*(b - v*c)                                    # E*-decay rate of the boosted family
    pmax = max(40.0, 40.0/max(bfl - abs(γ*(c - v*b)) - 1e-2, 5e-2))
    SE = SM = Sν = 0.0
    plo = -pmax
    prpan = (-pmax, -12.0, -2.0, 2.0, 12.0, pmax)
    qpan = (0.0, 2.0, 12.0, pmax)
    for a in 1:length(prpan)-1
        prlo = max(prpan[a], -pmax); prhi = min(prpan[a+1], pmax)
        prhi > prlo || continue
        ph = 0.5*(prhi - prlo); pm = 0.5*(prhi + prlo)
        for bq in 1:length(qpan)-1
            qlo, qhi = qpan[bq], min(qpan[bq+1], pmax)
            qhi > qlo || continue
            qh = 0.5*(qhi - qlo); qm = 0.5*(qhi + qlo)
            @inbounds for i in eachindex(_XS), j in eachindex(_XS)
                pr = pm + ph*_XS[i]; q = qm + qh*_XS[j]
                wq = ph*_WS[i]*qh*_WS[j]*q*2π
                E = sqrt(M*M + pr*pr + q*q)
                p2 = pr*pr + q*q
                DN = E + v*pr                            # w̃ denominator
                wt = (pr + v*E)/DN
                f = exp(-b*γ*(E + v*pr) + c*γ*(pr + v*E) + e*wt)
                lap = (2*E*E + M*M)/E^3                  # ∇²E
                # S_E, S_M: polynomial weights, closed forms
                SE += wq*f*(-ηD*M*γ*(p2/(E*E) + v*pr/E) + κ2*γ*lap)
                SM += wq*f*(-ηD*M*γ*(pr/E + v*p2/(E*E)) + κ2*γ*v*lap)
                # S_ν: w̃ = NUM/DN rational; gradients in (pr, q) closed form
                dNdp = 1 + v*pr/E; dNdq = v*q/E          # ∇(pr + vE)
                dDdp = pr/E + v;   dDdq = q/E            # ∇(E + v pr)
                w_p = (dNdp*DN - (pr + v*E)*dDdp)/DN^2
                w_q = (dNdq*DN - (pr + v*E)*dDdq)/DN^2
                gradNdotD = pr/E + v + v*p2/(E*E) + v*v*pr/E
                absD2 = p2/(E*E) + 2*v*pr/E + v*v
                lapN = v*lap; lapD = lap
                lapw = lapN/DN - 2*gradNdotD/DN^2 - (pr + v*E)*lapD/DN^2 +
                       2*(pr + v*E)*absD2/DN^3
                Sν += wq*f*(-ηD*M*(pr*w_p + q*w_q)/E + κ2*lapw)
            end
        end
    end
    (scale*SE, scale*SM, scale*Sν)
end

# ═══════════════════════ Right-hand side ═════════════════════════════════════════════════════════
@inline minmod(a, b) = (a*b <= 0) ? 0.0 : (abs(a) < abs(b) ? a : b)
@inline function _faces(u, k, N)
    iL  = clamp(k, 1, N);     iR  = clamp(k + 1, 1, N)
    iLL = clamp(k - 1, 1, N); iRR = clamp(k + 2, 1, N)
    sL = minmod(u[iL] - u[iLL], u[iR] - u[iL])
    sR = minmod(u[iR] - u[iL],  u[iRR] - u[iR])
    (u[iL] + 0.5*sL, u[iR] - 0.5*sR)
end

mutable struct M2Counters
    floored::Int
    fallback::Int
    eclamped::Int
    failed::Int
end
M2Counters() = M2Counters(0, 0, 0, 0)
@inline function _count!(cnt::M2Counters, flag::Int)
    flag == 1 && (cnt.floored += 1)
    flag == 2 && (cnt.eclamped += 1)
    flag == 3 && (cnt.fallback += 1)
    flag == 4 && (cnt.failed += 1)
    nothing
end

"""One RHS evaluation.  `U = (qN, QE, QM, qν)` = τr(N^τ, T^{ττ}, T^{τr}, N^r).
Reconstruct and dissipate the PHYSICAL densities, never the τr-weighted ones (the M1 r=0 lesson,
main2M1.jl:349-361); Rusanov λ = 1 (measured max|λ| = 0.984, m2_hyperbolicity.jl)."""
function compute_dUdt!(dU, U, τ::Float64, grid::M2Grid1D, bg::M2Background;
                       M::Float64, DsT, T_floor::Float64, cnt::M2Counters,
                       bC::Vector{Float64}, cC::Vector{Float64}, eC::Vector{Float64})
    Nr = length(grid.r); τs = max(τ, 1e-12)
    qN, QE, QM, qν = U
    Dc = zeros(Nr); Ec = zeros(Nr); Mc = zeros(Nr); Vc = zeros(Nr)
    @inbounds for i in 1:Nr
        jac = τs*max(grid.r[i], 1e-12)
        Dc[i] = qN[i]/jac; Ec[i] = QE[i]/jac; Mc[i] = QM[i]/jac; Vc[i] = qν[i]/jac
    end
    # ── cell-centre recoveries (warm-started from the previous call through bC/cC/eC) ──────────
    Pp_c = zeros(Nr); Kph_c = zeros(Nr); Kre_c = zeros(Nr); scl_c = zeros(Nr)
    okmask = falses(Nr)
    Threads.@threads :static for i in 1:Nr
        b, c, e, m, ok, flag = recover_m2(Dc[i], Ec[i], Mc[i], Vc[i], M;
                                          b0 = bC[i], c0 = cC[i], e0 = eC[i])
        bC[i] = b; cC[i] = c; eC[i] = e
        if m !== nothing && Dc[i] > M2_N_FLOOR
            s = Dc[i]/m.m1
            Pp_c[i] = s*m.mPp; Kph_c[i] = s*m.mKph; Kre_c[i] = s*m.mKre; scl_c[i] = s
            okmask[i] = true
        end
        flag != 0 && _count_locked!(cnt, flag)
    end
    for c in 1:4; fill!(dU[c], 0.0); end

    # ── fluxes ──────────────────────────────────────────────────────────────────────────────────
    F = zeros(4, Nr + 1)
    Threads.@threads :static for k in 0:Nr
        jacf = τs*max(grid.rF[k + 1], 0.0)
        aL, aR = _faces(Dc, k, Nr); bL, bR = _faces(Ec, k, Nr)
        cL, cR = _faces(Mc, k, Nr); vL, vR = _faces(Vc, k, Nr)
        iL = clamp(k, 1, Nr); iR = clamp(k + 1, 1, Nr)
        b1, c1, e1, mL, okL, fL = recover_m2(aL, bL, cL, vL, M; b0 = bC[iL], c0 = cC[iL], e0 = eC[iL])
        b2, c2, e2, mR, okR, fR = recover_m2(aR, bR, cR, vR, M; b0 = bC[iR], c0 = cC[iR], e0 = eC[iR])
        fL != 0 && _count_locked!(cnt, fL)
        fR != 0 && _count_locked!(cnt, fR)
        FL1 = FL2 = FL3 = FL4 = 0.0
        FR1 = FR2 = FR3 = FR4 = 0.0
        if mL !== nothing && aL > M2_N_FLOOR
            s = aL/mL.m1
            FL1 = s*mL.mn; FL2 = s*mL.mp; FL3 = s*mL.mTrr; FL4 = s*mL.mKrr
        end
        if mR !== nothing && aR > M2_N_FLOOR
            s = aR/mR.m1
            FR1 = s*mR.mn; FR2 = s*mR.mp; FR3 = s*mR.mTrr; FR4 = s*mR.mKrr
        end
        F[1, k + 1] = jacf*(0.5*(FL1 + FR1) - 0.5*(aR - aL))
        F[2, k + 1] = jacf*(0.5*(FL2 + FR2) - 0.5*(bR - bL))
        F[3, k + 1] = jacf*(0.5*(FL3 + FR3) - 0.5*(cR - cL))
        F[4, k + 1] = jacf*(0.5*(FL4 + FR4) - 0.5*(vR - vL))
    end
    @inbounds for k in 1:Nr
        for c in 1:4
            dU[c][k] -= (F[c, k + 1] - F[c, k])/grid.dr[k]
        end
    end

    # ── geometry + drag sources ─────────────────────────────────────────────────────────────────
    Threads.@threads :static for i in 1:Nr
        okmask[i] || continue
        r = grid.r[i]
        dU[2][i] -= r*Pp_c[i]                            # −rP⊥   (Γ^τ_{ηη})
        dU[3][i] += τs*Pp_c[i]                           # +τP⊥   (Γ^r_{φφ})
        dU[4][i] += τs*Kph_c[i] + r*Kre_c[i]             # +τK^{φφ} + rK^{rηη}  (the ν̂ geometry)
        Dc[i] > M2_N_FLOOR || continue
        T = max(bg_T(bg, τ, r), T_floor)
        uτ, ur = _u_from_ur(bg_ur(bg, τ, r)); v = ur/uτ
        ηD = eta_drag(T, M, effective_DsT(T, DsT))
        SE, SM, Sν = sources_m2(bC[i], cC[i], eC[i], scl_c[i], v, T, ηD, M)
        jac = τs*max(r, 1e-12)
        dU[2][i] += jac*SE
        dU[3][i] += jac*SM
        dU[4][i] += jac*Sν
    end
    return nothing
end

const _CNT_LOCK = ReentrantLock()
@inline function _count_locked!(cnt::M2Counters, flag::Int)
    lock(_CNT_LOCK) do
        _count!(cnt, flag)
    end
end

# ═══════════════════════ Driver ══════════════════════════════════════════════════════════════════
"""
    solve_M2(grid, α0, νr0, τ0, τf, bg; ...) -> Dict

SSP-RK2.  Output contract = main2IS2/M1's exactly (`r_grid`/`t_grid`, (Nr, Nt) arrays `n`, `nur`,
`alpha`) plus M2's `Tprime` (= 1/√(b²−c²), the e = 0 notion), `w`, `Ntau`, `e`, and a
`diagnostics` Dict counting cone floors, e-clamps, fallbacks and failures.
"""
function solve_M2(grid::M2Grid1D, α0::Vector{Float64}, νr0::Vector{Float64},
                  τ0::Float64, τf::Float64, bg::M2Background;
                  CFL::Float64 = 0.25, save_dt::Float64 = 0.1, DsT = 0.1163,
                  T_floor::Float64 = 1e-6, eos = LatticeHRGEOS(canon_factor = 1.0),
                  log_every::Int = 100, Tprime0 = nothing)
    Nr = length(grid.r); M = hq_mass(eos)
    _build_h_table!(M)

    # ── IC: the e = 0 boosted Jüttner carrying exactly (n, ν^r) at the fluid temperature ────────
    # (interior to the joint cone BY CONSTRUCTION: qν is set to the family's own N^r = n′u′^r)
    qN = zeros(Nr); QE = zeros(Nr); QM = zeros(Nr); qν = zeros(Nr)
    bC = fill(NaN, Nr); cC = zeros(Nr); eC = zeros(Nr)
    for i in 1:Nr
        r = grid.r[i]; T = max(bg_T(bg, τ0, r), T_floor)
        _, n_eq, _ = eos_Pne(T, α0[i]*T, eos)
        n = max(n_eq, 0.0)
        uτ, ur = _u_from_ur(bg_ur(bg, τ0, r))
        w = n > M2_N_FLOOR ? clamp(νr0[i]/(n*uτ), -0.999999, 0.999999) : 0.0
        γw = 1/sqrt(1 - w*w)
        upτ = γw*(uτ + w*ur); upr = γw*(ur + w*uτ)
        np = n/γw
        Tp0 = Tprime0 === nothing ? T : Float64(Tprime0 isa Function ? Tprime0(r) : Tprime0)
        hh = np*hpp(Tp0, M)
        jac = τ0*max(r, 1e-12)
        qN[i] = jac*np*upτ
        QE[i] = jac*(hh*upτ*upτ - np*Tp0)
        QM[i] = jac*(hh*upτ*upr)
        qν[i] = jac*np*upr                                # N^r of the e = 0 state, EXACT
    end

    U  = (qN, QE, QM, qν)
    U1 = ntuple(c -> similar(U[c]), 4)
    dU = ntuple(c -> similar(U[c]), 4)
    cnt = M2Counters()
    τ = τ0; step = 0
    dr_min = minimum(grid.dr)
    snaps_τ = Float64[]; snaps = Vector{NTuple{9,Vector{Float64}}}()
    next_save = τ0

    while τ < τf - 1e-12
        dt = min(CFL*dr_min, τf - τ)
        compute_dUdt!(dU, U, τ, grid, bg; M = M, DsT = DsT, T_floor = T_floor, cnt = cnt,
                      bC = bC, cC = cC, eC = eC)
        for c in 1:4; @. U1[c] = U[c] + dt*dU[c]; end
        compute_dUdt!(dU, U1, τ + dt, grid, bg; M = M, DsT = DsT, T_floor = T_floor, cnt = cnt,
                      bC = bC, cC = cC, eC = eC)
        for c in 1:4; @. U[c] = 0.5*(U[c] + U1[c] + dt*dU[c]); end
        τ += dt; step += 1

        if τ >= next_save - 1e-12 || τ >= τf - 1e-12
            push!(snaps_τ, τ); push!(snaps, _snapshot(U, τ, grid, bg, M, T_floor, eos, bC, cC, eC))
            next_save += save_dt
        end
        log_every > 0 && step % log_every == 0 &&
            @info @sprintf("M2 τ=%.3f dt=%.4f steps=%d floored=%d eclamp=%d fb=%d fail=%d",
                           τ, dt, step, cnt.floored, cnt.eclamped, cnt.fallback, cnt.failed)
    end

    Nt = length(snaps_τ)
    grab(k) = (A = zeros(Nr, Nt); for j in 1:Nt, i in 1:Nr; A[i, j] = snaps[j][k][i]; end; A)
    return Dict{String,Any}(
        "r_grid" => copy(grid.r), "t_grid" => collect(snaps_τ),
        "n" => grab(1), "nur" => grab(2), "alpha" => grab(3),
        "Tprime" => grab(4), "w" => grab(5), "Ntau" => grab(6), "e" => grab(7),
        "b" => grab(8), "c" => grab(9),
        "diagnostics" => Dict("realizability_floor" => cnt.floored, "e_clamped" => cnt.eclamped,
                              "fallback" => cnt.fallback, "failed" => cnt.failed, "steps" => step),
    )
end

"""Fluid-frame observables from the conserved state (uses the stored cell multipliers)."""
function _snapshot(U, τ, grid::M2Grid1D, bg::M2Background, M, T_floor, eos, bC, cC, eC)
    Nr = length(grid.r); τs = max(τ, 1e-12)
    n_a = zeros(Nr); ν_a = zeros(Nr); α_a = zeros(Nr); T_a = zeros(Nr); w_a = zeros(Nr)
    Nt_a = zeros(Nr); e_a = zeros(Nr); b_a = zeros(Nr); c_a = zeros(Nr)
    qN, QE, QM, qν = U
    @inbounds for i in 1:Nr
        r = grid.r[i]; jac = τs*max(r, 1e-12)
        D = qN[i]/jac; ν̂ = qν[i]/jac
        T = max(bg_T(bg, τ, r), T_floor)
        uτ, ur = _u_from_ur(bg_ur(bg, τ, r)); v = ur/uτ
        VN = D > M2_N_FLOOR ? clamp(ν̂/D, -0.999999, 0.999999) : 0.0   # density-frame velocity
        w = clamp((VN - v)/(1 - VN*v), -0.999999, 0.999999)
        γN = 1/sqrt(max(1 - VN*VN, 1e-300))
        # u·N with u the FLUID flow: n = uτ·N^τ − ur·N^r
        n = max(uτ*D - ur*ν̂, 0.0)
        _, n_eq, _ = eos_Pne(T, 0.0, eos)
        n_a[i] = n
        ν_a[i] = uτ*ν̂ - ur*D                             # −Δ^r_μ N^μ = the fluid-frame current
        α_a[i] = n_eq > TINY && n > M2_N_FLOOR ? log(n/n_eq) : -700.0
        bb, cc = bC[i], cC[i]
        T_a[i] = (isfinite(bb) && bb > abs(cc)) ? 1/sqrt(bb*bb - cc*cc) : M2_TP_LO
        w_a[i] = w
        Nt_a[i] = D
        e_a[i] = eC[i]
        b_a[i] = bb; c_a[i] = cc
    end
    (n_a, ν_a, α_a, T_a, w_a, Nt_a, e_a, b_a, c_a)
end

function run_static_M2_test(; background_file::String, DsT = 0.1163,
                            τ0::Float64 = 0.4, τfinal::Float64 = 8.0,
                            Nr::Int = 300, rmax::Float64 = 25.0, CFL::Float64 = 0.25,
                            dump_dt::Float64 = 0.1, T_floor::Float64 = 1e-6,
                            init_mode::Symbol = :auto,
                            n_profile = r -> exp(-r^2/(2.0*4.0^2)), nur_profile = nothing,
                            eos = LatticeHRGEOS(canon_factor = 1.0), log_every::Int = 100)
    bg = load_M2_background(background_file)
    grid = M2Grid1D(Nr, min(rmax, last(bg.r_grid)))
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
    solve_M2(grid, α, νr, τ0, τfinal, bg; CFL = CFL, save_dt = dump_dt, DsT = DsT,
             T_floor = T_floor, eos = eos, log_every = log_every)
end

end # module hydro_current_M2
