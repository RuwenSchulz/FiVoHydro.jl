# ==============================================================================
# test/analytic_referees.jl — closed-form and semi-analytic REFEREES for FiVo.
#
# Pure functions of (τ, r) and coefficients: no grid, no solver state, no FiVo
# numerics. Each referee is written once and judges every solver that can run its
# state — the 1+1D bulk solver, the 1+1D charm IS2 solver and the 2+1D solver —
# which is what makes a disagreement between two of them attributable.
#
#   bjorken_background     ideal Bjorken with a conserved charge, any EOS
#   bjorken_dnmr           viscous Bjorken, the full 0+1D DNMR set (δ_ππ, τ_ππ,
#                          λ_πΠ, δ_ΠΠ, λ_Ππ each explicit)
#   gubser_viscous         viscous Gubser flow (semi-analytic ODE in ρ)
#   diffusion_mode         the radial diffusion mode δn = A J₀(kr), ν^r = B J₁(kr)
#                          on a uniform Bjorken background, with the consistent
#                          first moment's expansion and D ln h terms switchable
#   interp1                linear lookup on a uniform grid
#
# Sign conventions: mostly-PLUS metric, π_NS = −2ησ (the codes' convention), and
# φ ≡ −τ²π^{ηη} = −π^η_η ≥ 0 for an expanding fluid (the shear-pressure difference).
# Each function documents its derivation; the 0+1D DNMR reduction was checked
# against Jaiswal, Ryblewski & Strickland, PRC 90 (2014) 044908, Eqs. (8)-(9).
# ==============================================================================

using SpecialFunctions: besselj0, besselj1, besselkx

const HBARC_REF = 0.1973269804      # GeV fm

"""RK4 for y' = f(t, y) with fixed step, y a tuple. Returns (ts, ys)."""
function rk4(f, t0, y0::Tuple, t1; n::Int = 20_000)
    h = (t1 - t0)/n
    ts = Vector{Float64}(undef, n+1); ys = Vector{typeof(y0)}(undef, n+1)
    ts[1] = t0; ys[1] = y0; t = t0; y = y0
    for i in 1:n
        k1 = f(t, y)
        k2 = f(t + h/2, y .+ (h/2) .* k1)
        k3 = f(t + h/2, y .+ (h/2) .* k2)
        k4 = f(t + h, y .+ h .* k3)
        y = y .+ (h/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
        t = t0 + i*h
        ts[i+1] = t; ys[i+1] = y
    end
    return ts, ys
end

"""Linear lookup of samples `v` on the uniform grid `xs`."""
function interp1(xs::AbstractVector, v::AbstractVector)
    lo = xs[1]; h = xs[2] - xs[1]; n = length(xs)
    return function (x)
        t = (x - lo)/h
        j = clamp(floor(Int, t) + 1, 1, n-1)
        w = t - (j - 1)
        (1 - w)*v[j] + w*v[j+1]
    end
end

# ------------------------------------------------------------------------------
# Bjorken: ideal background with a conserved charge
# ------------------------------------------------------------------------------
"""
    bjorken_background(eos_Pne, T0, α0, τ0, τ1; n) -> (τs, T(τ), α(τ))

Ideal boost-invariant flow of a fluid with a conserved charge:

    d(nτ)/dτ = 0 ,    de/dτ = −(e + P)/τ ,

integrated in the primitive pair (T, α) with the Jacobian of `eos_Pne(T, μ)`
(central differences). `eos_Pne` is the solver's EOS — the EOS is gated on its
own (test_eos_consistency.jl); this referee checks the time evolution.
"""
function bjorken_background(eos_Pne, T0, α0, τ0, τ1; n::Int = 20_000)
    function thermo(T, α)
        P, nn, e = eos_Pne(T, α*T)
        return P, nn, e
    end
    function f(τ, y)
        T, α = y
        P, nn, e = thermo(T, α)
        hT = 1e-6*T; hα = 1e-6
        _, npT, epT = thermo(T + hT, α); _, nmT, emT = thermo(T - hT, α)
        _, npα, epα = thermo(T, α + hα); _, nmα, emα = thermo(T, α - hα)
        nT = (npT - nmT)/(2hT); eT = (epT - emT)/(2hT)
        nα = (npα - nmα)/(2hα); eα = (epα - emα)/(2hα)
        # [nT nα; eT eα] [dT; dα] = [−n/τ; −(e+P)/τ]
        b1 = -nn/τ; b2 = -(e + P)/τ
        det = nT*eα - nα*eT
        return ((b1*eα - nα*b2)/det, (nT*b2 - b1*eT)/det)
    end
    τs, ys = rk4(f, τ0, (T0, α0), τ1; n)
    return τs, interp1(τs, first.(ys)), interp1(τs, last.(ys))
end

# ------------------------------------------------------------------------------
# Viscous Bjorken — the full 0+1D DNMR set
# ------------------------------------------------------------------------------
"""
    bjorken_dnmr(; eos_eP, Tof_e, η, τπ, ζ, τΠ, δππ = 0, τππ = 0, λπΠ = 0,
                   δΠΠ = 0, λΠπ = 0, e0, φ0 = 0, Π0 = 0, τ0, τ1, n)
        -> (τs, e(τ), φ(τ), Π(τ))

Boost-invariant, transversely uniform, charge-free viscous flow. With
φ = −π^η_η and every coefficient a function of the energy density (through T):

    de/dτ = −(e + P + Π − φ)/τ
    τ_π dφ/dτ = −φ + (4/3)η/τ − (δ_ππ + τ_ππ/3) φ/τ + (2/3) λ_πΠ Π/τ
    τ_Π dΠ/dτ = −Π − ζ/τ − δ_ΠΠ Π/τ + λ_Ππ φ/τ

This is the mostly-plus reduction of τ_π Δ Dπ + π = −2ησ − δ_ππθπ − τ_ππ π^⟨μ_λσ^ν⟩λ
− λ_πΠΠσ and τ_Π DΠ + Π = −ζθ − δ_ΠΠΠθ − λ_Ππ π:σ, using σ^η_η = 2/(3τ),
θ = 1/τ, π:σ = −φ/τ on Bjorken (JRS 2014, Eqs. 8-9, with their π ≡ φ).

`η(T)`, `τπ(T)`, `ζ(T)`, `τΠ(T)` are the solver's transport coefficients evaluated
at T = `Tof_e(e)`; `δππ` etc. are RATIOS to the relaxation time (the solver's
`*_factor` knobs).
"""
function bjorken_dnmr(; eos_eP, Tof_e, η, τπ, ζ = T -> 0.0, τΠ = T -> 1.0,
                        δππ = 0.0, τππ = 0.0, λπΠ = 0.0, δΠΠ = 0.0, λΠπ = 0.0,
                        e0, φ0 = 0.0, Π0 = 0.0, τ0, τ1, n::Int = 200_000)
    function f(τ, y)
        e, φ, Π = y
        T = Tof_e(e); _, P = eos_eP(T)
        tp = τπ(T); tb = τΠ(T)
        de = -(e + P + Π - φ)/τ
        dφ = tp > 0 ? (-φ + (4/3)*η(T)/τ - (δππ + τππ/3)*tp*φ/τ + (2/3)*λπΠ*tp*Π/τ)/tp : 0.0
        dΠ = tb > 0 && ζ(T) > 0 ? (-Π - ζ(T)/τ - δΠΠ*tb*Π/τ + λΠπ*tb*φ/τ)/tb : 0.0
        return (de, dφ, dΠ)
    end
    τs, ys = rk4(f, τ0, (e0, φ0, Π0), τ1; n)
    return τs, interp1(τs, getindex.(ys, 1)), interp1(τs, getindex.(ys, 2)), interp1(τs, getindex.(ys, 3))
end

# ------------------------------------------------------------------------------
# Viscous Gubser flow (conformal) — the semi-analytic referee of gate G1v
# ------------------------------------------------------------------------------
"""
    gubser_viscous(ηs; TH0 = 0.6, ρmin = -6, ρmax = 2, n = 400_000, seed_ns = true)
        -> (ρs, T̂(ρ), π̄(ρ))

Weyl-rescaled to de Sitter the fluid is at rest; with π̄ = π^η_η/(e + P) and the IS
shear equation with δ_ππ = (4/3)τ_π (derivation: test_gubser_viscous2d.jl):

    dT̂/dρ = −(2/3) T̂ tanh ρ + (1/3) T̂ π̄ tanh ρ
    dπ̄/dρ = −(4/3) π̄² tanh ρ − π̄/τ̂_π + (4/15) tanh ρ ,    τ̂_π = 5 (η/s)/T̂ ,

T̂ in fm⁻¹ units (T̂/ħc). Integrated FORWARD in ρ only (backward is exponentially
unstable — the relaxation anti-damps). Map to Milne: T(τ, r) = T̂(ρ)/τ,
ρ = asinh(−(1 − q²τ² + q²r²)/(2qτ)), and π^η_η = π̄ (e + P).
"""
function gubser_viscous(ηs; TH0 = 0.6, ρmin = -6.0, ρmax = 2.0, n::Int = 400_000, seed_ns = true)
    invfmGeV = 1/HBARC_REF
    f(ρ, y) = begin
        T, p = y; th = tanh(ρ)
        τp = ηs > 0 ? 5*ηs/(max(T, 1e-12)*invfmGeV) : 0.0
        (-(2/3)*T*th + (1/3)*T*p*th, ηs > 0 ? -(4/3)*p*p*th - p/τp + (4/15)*th : 0.0)
    end
    T1 = TH0*cosh(ρmin)^(-2/3)
    p1 = (ηs > 0 && seed_ns) ? (4/3)*ηs*tanh(ρmin)/(T1*invfmGeV) : 0.0
    ρs, ys = rk4(f, ρmin, (T1, p1), ρmax; n)
    return ρs, interp1(ρs, first.(ys)), interp1(ρs, last.(ys))
end
gubser_rho(τ, r; q = 1.0) = asinh(-(1 - q^2*τ^2 + q^2*r^2)/(2q*τ))
"""Gubser u^r (conformal, any viscosity): 2q²τr / sqrt(1 + 2q²(τ² + r²) + q⁴(τ² − r²)²)."""
gubser_ur(τ, r; q = 1.0) = 2q^2*τ*r / sqrt(1 + 2q^2*(τ^2 + r^2) + q^4*(τ^2 - r^2)^2)

# ------------------------------------------------------------------------------
# The radial diffusion mode on a Bjorken background
# ------------------------------------------------------------------------------
"""
    diffusion_mode(; k, DsT, m, Tbar, τ0, τ1, A0, B0 = 0.0,
                     expansion = false, dlnh = false, δnn = 0.0, n) -> (τs, A(τ), B(τ))

A small charge perturbation on a transversely uniform, charge-carrying Bjorken
background (T̄(τ) given — `bjorken_background`), with u^r = 0 exactly:

    δn(τ, r) = A(τ) J₀(kr) ,     ν^r(τ, r) = B(τ) J₁(kr) .

J₀, J₁ are the eigenfunctions of the radial problem — ∂_r J₀ = −kJ₁ and
(1/r)∂_r(rJ₁) = kJ₀ — so the linearised equations close on (A, B) EXACTLY:

    charge   ∂_τ(τ δn) + τ (1/r)∂_r(r ν^r) = 0          ⇒  dA/dτ = −A/τ − kB
    current  τ_n ∂_τν + ν = −κ ∂_rδα − b ν ,  κ = D_s n̄ ,  δα = δn/n̄
                                                        ⇒  τ_n dB/dτ = D_s k A − (1 + b) B

(δα = δn/n̄ because n ∝ e^α at fixed T for the Boltzmann charm.) `b` collects the
ν-proportional terms of the consistent first moment on this background, each
switchable:  τ_n θ (θ = 1/τ, `expansion`) + (D_s/T) h′ dT̄/dτ (`dlnh`) — the
pressure-gradient and inertial terms vanish (∇T = 0, a = 0), and so does ν·∇u —
plus a δ_nn θ damping. Coefficients in CLOSED FORM, independent of the solvers:
D_s = D_sT ħc/T̄ [fm], τ_n = D_s z K₃/K₂, h = m K₃/K₂, z = m/T̄.

The shipped row (`expansion = dlnh = false`) is the pure telegraph mode:
without expansion it has damping rates ω = (1 ± √(1 − 4k²D_sτ_n))/(2τ_n).
"""
function diffusion_mode(; k, DsT, m, Tbar, τ0, τ1, A0, B0 = 0.0,
                          expansion::Bool = false, dlnh::Bool = false, δnn::Float64 = 0.0,
                          n::Int = 40_000)
    function coeffs(τ)
        T = Tbar(τ); z = m/T
        K1 = besselkx(1, z); K2 = besselkx(2, z); K3 = besselkx(3, z); K4 = besselkx(4, z)
        Ds = DsT*HBARC_REF/T
        τn = Ds*z*K3/K2
        hp = (m*z/T)*((K2 + K4)*K2 - (K1 + K3)*K3)/(2K2^2)     # dh/dT, h = m K3/K2
        dT = (Tbar(τ + 1e-5) - Tbar(τ - 1e-5))/2e-5
        b = (expansion ? τn/τ : 0.0) + (dlnh ? (Ds/T)*hp*dT : 0.0) + δnn*τn/τ
        return Ds, τn, b
    end
    function f(τ, y)
        A, B = y
        Ds, τn, b = coeffs(τ)
        return (-A/τ - k*B, (Ds*k*A - (1 + b)*B)/τn)
    end
    τs, ys = rk4(f, τ0, (A0, B0), τ1; n)
    return τs, interp1(τs, first.(ys)), interp1(τs, last.(ys))
end

"""Least-squares projection amplitude of samples `v(r)` onto `basis(r)` with weight r."""
function project_mode(rs, v, basis)
    num = 0.0; den = 0.0
    for (r, x) in zip(rs, v)
        b = basis(r); num += r*b*x; den += r*b*b
    end
    return num/den
end
