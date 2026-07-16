#!/usr/bin/env julia
# ==============================================================================
# mainBDNK_causal.jl — GENUINE causal BDNK charge diffusion (production driver)
#
# Replaces the parabolic-in-disguise schemes of mainBDNK.jl / main2BDNK.jl (which set ν^r
# algebraically and drop the frame τ-component, see Projects/FiVoBenchmark/CAUSAL_HYDRO_AUDIT.md
# findings B-2/B-2R) with a self-consistent CAUSAL HYPERBOLIC evolution.
#
# Physics (Tex/ModePaper1/bdnk_equations.wl): BDNK keeps the diffusion current's frame τ-component
# in the CONSERVED charge density (it intentionally breaks Landau orthogonality to buy causality):
#     J^μ = n u^μ + ν^μ,   ν^μ = −κ Δ^{μν}∂_ν α + ε_ν u^μ (u^ν ∂_ν α)
# In Milne (mostly-plus, boost-invariant + rot-sym; ∂_τ,∂_r only), with v_sig = √(κ/ε_ν) ≤ c:
#     ν^τ = A β + B α_r,   ν^r = B β + C α_r,    β ≡ ∂_τ α,  α_r ≡ ∂_r α
#     A = ε_ν(u^τ)² − κ(u^r)²,  B = u^τ u^r (ε_ν − κ),  C = ε_ν(u^r)² − κ(u^τ)²
#
# CONSERVATIVE scheme (manifestly causal, reduces to the verified telegraph prototype on static bg —
# bench_bdnk_causal_signal.jl): evolve  q ≡ τ r J^τ  and  α ; reconstruct β algebraically:
#     β = ( q/(τ r) − n u^τ − B α_r ) / A         [from J^τ = n u^τ + ν^τ]
#     ∂_τ α = β
#     ∂_τ q = −∂_r( τ r J^r ),   J^r = n u^r + B β + C α_r
# Time integration: SSPRK3 (stable on the imaginary axis — SSPRK2 is NOT; see the prototype).
# CFL is HYPERBOLIC: dt ~ CFL·dr/max(|v|, v_sig), not parabolic dr²/κ.
#
# Background T(τ,r), u^r(τ,r) are external (probe limit — relevant for charm); α is DYNAMICAL.
# Supports analytic callable backgrounds (for tests) or a loaded BackgroundFields (.jld2).
# ==============================================================================

include(joinpath(@__DIR__, "main.jl"))

module bdnk_causal

import ..hydro
import ..hydro: eos_Pne, LatticeHRGEOS, ConformalHQEOS, fmGeV, hq_mass, TINY, T_MIN
using Printf

# ----------------------------------------------------------------------------
# Background: callable T(τ,r), ur(τ,r); α0(r) initial fugacity profile.
# ----------------------------------------------------------------------------
struct CausalBG
    T::Function     # (τ,r) -> T  [GeV]
    ur::Function    # (τ,r) -> u^r
    α0::Function    # (r)   -> α(τ0,r)  initial fugacity
end

"""Static uniform background (u^r=0, T=const) — the telegraph-prototype regime."""
static_bg(; T0::Float64=0.3, α0::Float64=0.0) =
    CausalBG((τ,r)->T0, (τ,r)->0.0, r->α0)

"""Bjorken-like background: T ∝ τ^{-1/3} cooling, u^r=0 (pure longitudinal)."""
bjorken_bg(; T0::Float64=0.3, τ0::Float64=1.0, α0::Float64=0.0) =
    CausalBG((τ,r)->T0*(τ0/τ)^(1/3), (τ,r)->0.0, r->α0)

# ----------------------------------------------------------------------------
# Transport / thermodynamics
# ----------------------------------------------------------------------------
@inline n_of(α, T, eos) = eos_Pne(T, α*T, eos)[2]

"""χ_α = ∂n/∂α at fixed T (central finite difference)."""
@inline function chi_alpha(α, T, eos; h=1e-4)
    (n_of(α+h, T, eos) - n_of(α-h, T, eos)) / (2h)
end

@inline kappa_of(T, n, DsT) = DsT * max(n, 0.0) / max(T, T_MIN) / fmGeV

"""Local ε_ν ≥ κ (causal).  :kappa ⇒ ε_ν=κ (v_sig=c, minimal causal);
   :is_match ⇒ ε_ν=χ_α·τ_n (FP/IS gap), floored at κ to guarantee causality."""
@inline function eps_nu_of(T, α, n, κ, DsT, eos, mode::Symbol)
    if mode === :is_match
        z = hq_mass(eos)/max(T, T_MIN)
        τn = (DsT/ max(T,T_MIN)) * z * _K3overK2(z) / fmGeV   # FP Bessel τ_n (GeV→fm)
        χ  = chi_alpha(α, T, eos)
        return max(χ*τn, κ)
    else
        return κ          # :kappa — minimal causal
    end
end

@inline function _K3overK2(z)
    z <= 0 && return 1.0
    k2 = hydro.SpecialFunctions.besselkx(2, z)
    k3 = hydro.SpecialFunctions.besselkx(3, z)
    return k2 > 0 ? k3/k2 : 1.0
end

# ----------------------------------------------------------------------------
# Grid (interior cells 1..Nr, uniform; 1 ghost handled via BCs)
# ----------------------------------------------------------------------------
struct CausalGrid
    r::Vector{Float64}    # cell centers
    rF::Vector{Float64}   # faces (Nr+1)
    dr::Float64
end
function make_causal_grid(Nr::Int; rmax::Float64)
    dr = rmax/Nr
    r  = [(i-0.5)*dr for i in 1:Nr]
    rF = [(i-1)*dr   for i in 1:(Nr+1)]
    return CausalGrid(r, rF, dr)
end

# ----------------------------------------------------------------------------
# Per-cell BDNK coefficients (A,B,C) and primitives
# ----------------------------------------------------------------------------
@inline function cell_coeffs(α, T, ur, DsT, eos, mode; eps_factor::Float64=1.0)
    uτ = sqrt(1+ur^2)
    n  = n_of(α, T, eos)
    κ  = kappa_of(T, n, DsT)
    εν = max(eps_factor * eps_nu_of(T, α, n, κ, DsT, eos, mode), κ)   # ε_ν≥κ ⇒ causal
    A  = εν*uτ^2 - κ*ur^2
    B  = uτ*ur*(εν - κ)
    C  = εν*ur^2 - κ*uτ^2
    return (n=n, uτ=uτ, κ=κ, εν=εν, A=A, B=B, C=C)
end

# measure √-g: Milne boost-inv+rotsym ⇒ τr ; Cartesian (validation) ⇒ 1
@inline meas(geom::Symbol, τ, r) = geom === :cartesian ? 1.0 : τ*r

# ----------------------------------------------------------------------------
# RHS of (q, α): returns (dq, dα).  q = m J^τ,  m = measure (τr Milne / 1 Cartesian).
# ----------------------------------------------------------------------------
function rhs!(dq, dα, q, α, τ, grid::CausalGrid, bg::CausalBG; DsT, eos, mode, geom::Symbol=:milne, eps_factor::Float64=1.0)
    Nr = length(grid.r); dr = grid.dr
    # cell primitives
    β  = Vector{Float64}(undef, Nr)
    Jr = Vector{Float64}(undef, Nr)
    # ∂_r α (central, axis reflective α even, outer zero-gradient)
    αr = Vector{Float64}(undef, Nr)
    @inbounds for i in 1:Nr
        αm = i==1  ? α[1]  : α[i-1]    # axis: even reflection (α_ghost=α[1])
        αp = i==Nr ? α[Nr] : α[i+1]    # outer: zero-gradient
        αr[i] = (αp - αm) / (2dr)
    end
    @inbounds for i in 1:Nr
        T  = max(bg.T(τ, grid.r[i]), T_MIN)
        ur = bg.ur(τ, grid.r[i])
        c  = cell_coeffs(α[i], T, ur, DsT, eos, mode; eps_factor=eps_factor)
        # reconstruct β from conserved q: q/m = n uτ + A β + B αr
        Jτ = q[i] / meas(geom, τ, grid.r[i])
        β[i]  = (Jτ - c.n*c.uτ - c.B*αr[i]) / c.A
        Jr[i] = c.n*ur + c.B*β[i] + c.C*αr[i]
    end
    # face fluxes F = m J^r (m=measure at face)
    F = Vector{Float64}(undef, Nr+1)
    F[1] = 0.0                                   # axis: zero flux (regularity)
    @inbounds for f in 2:Nr
        Jr_f = 0.5*(Jr[f-1] + Jr[f])             # centered face flux
        F[f] = meas(geom, τ, grid.rF[f]) * Jr_f
    end
    # outer face: outflow (use last cell J^r)
    F[Nr+1] = meas(geom, τ, grid.rF[Nr+1]) * Jr[Nr]
    @inbounds for i in 1:Nr
        dq[i] = -(F[i+1] - F[i]) / dr
        dα[i] = β[i]
    end
    return nothing
end

# ----------------------------------------------------------------------------
# CFL: hyperbolic dt ~ CFL·dr/max(|v|,v_sig) and proper-time cap.
# ----------------------------------------------------------------------------
function compute_dt(α, τ, grid::CausalGrid, bg::CausalBG; DsT, eos, mode, CFL, CFLτ, eps_factor::Float64=1.0)
    Nr = length(grid.r); amax = 1e-8
    @inbounds for i in 1:Nr
        T  = max(bg.T(τ, grid.r[i]), T_MIN); ur = bg.ur(τ, grid.r[i])
        c  = cell_coeffs(α[i], T, ur, DsT, eos, mode; eps_factor=eps_factor)
        vsig = sqrt(max(c.κ/c.εν, 0.0))          # ≤ 1 by construction (ε_ν ≥ κ)
        v    = ur/sqrt(1+ur^2)
        # lab-frame boosted signal speed
        λ = (abs(v) + vsig) / (1 + abs(v)*vsig + TINY)
        amax = max(amax, λ)
    end
    return min(CFL*grid.dr/amax, CFLτ*τ)
end

# ----------------------------------------------------------------------------
# SSPRK3 driver.  Returns (τs, αs, qs) snapshots + signal-speed diagnostic.
# ----------------------------------------------------------------------------
function run_bdnk_causal(; Nr::Int=400, rmax::Float64=20.0, τ0::Float64=1.0, τfinal::Float64=5.0,
                          DsT::Float64=0.24, eos=ConformalHQEOS(g_eff=40.0, m_hq=1.5, g_hq=6.0),
                          mode::Symbol=:kappa, bg::CausalBG=static_bg(),
                          α_init::Union{Nothing,Function}=nothing, geom::Symbol=:milne,
                          eps_factor::Float64=1.0,
                          CFL::Float64=0.3, CFLτ::Float64=0.05, save_dt::Float64=0.25,
                          verbose::Bool=false)
    grid = make_causal_grid(Nr; rmax=rmax)
    αinit = α_init === nothing ? bg.α0 : α_init
    α = [αinit(r) for r in grid.r]
    # q0 = m J^τ with J^τ = n u^τ + ν^τ; at τ0 assume β=0 (equilibrium start) ⇒ ν^τ = B α_r
    q = Vector{Float64}(undef, Nr)
    @inbounds for i in 1:Nr
        T = max(bg.T(τ0, grid.r[i]), T_MIN); ur = bg.ur(τ0, grid.r[i])
        c = cell_coeffs(α[i], T, ur, DsT, eos, mode; eps_factor=eps_factor)
        αm = i==1 ? α[1] : α[i-1]; αp = i==Nr ? α[Nr] : α[i+1]
        αr = (αp-αm)/(2grid.dr)
        q[i] = meas(geom, τ0, grid.r[i]) * (c.n*c.uτ + c.B*αr)   # β=0 start
    end

    τs=[τ0]; αs=[copy(α)]; qs=[copy(q)]
    dq=similar(q); dα=similar(α); q1=similar(q); α1=similar(α); q2=similar(q); α2=similar(α)
    τ=τ0; next=τ0+save_dt
    while τ < τfinal - 1e-12
        dt = min(compute_dt(α, τ, grid, bg; DsT=DsT, eos=eos, mode=mode, CFL=CFL, CFLτ=CFLτ, eps_factor=eps_factor), τfinal-τ)
        # SSPRK3
        rhs!(dq, dα, q, α, τ, grid, bg; DsT=DsT, eos=eos, mode=mode, geom=geom, eps_factor=eps_factor)
        @. q1 = q + dt*dq;  @. α1 = α + dt*dα
        rhs!(dq, dα, q1, α1, τ+dt, grid, bg; DsT=DsT, eos=eos, mode=mode, geom=geom, eps_factor=eps_factor)
        @. q2 = 0.75q + 0.25*(q1 + dt*dq);  @. α2 = 0.75α + 0.25*(α1 + dt*dα)
        rhs!(dq, dα, q2, α2, τ+dt/2, grid, bg; DsT=DsT, eos=eos, mode=mode, geom=geom, eps_factor=eps_factor)
        @. q = q/3 + (2/3)*(q2 + dt*dq);  @. α = α/3 + (2/3)*(α2 + dt*dα)
        τ += dt
        if τ >= next-1e-12
            push!(τs, τ); push!(αs, copy(α)); push!(qs, copy(q)); next += save_dt
        end
        verbose && @info "bdnk_causal" τ=τ dt=dt
    end
    if τs[end] != τ                                  # always save the final state
        push!(τs, τ); push!(αs, copy(α)); push!(qs, copy(q))
    end
    return (τs=τs, αs=αs, qs=qs, grid=grid)
end

end # module bdnk_causal
