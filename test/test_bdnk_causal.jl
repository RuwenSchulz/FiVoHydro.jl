#!/usr/bin/env julia
# test_bdnk_causal.jl — regression test for the GENUINE causal BDNK production driver
# (mainBDNK_causal.jl). Run as an isolated subprocess from runtests.jl. Exits 0 on pass.
# Asserts: flat-α fixed point (Cartesian), Milne charge conservation, and causal support.

include(joinpath(@__DIR__, "..", "mainBDNK_causal.jl"))
using .bdnk_causal
const BC = bdnk_causal

allpass = true
fail(msg) = (global allpass; allpass = false; println("FAIL: ", msg))

bg = BC.static_bg(T0=0.3, α0=0.0)

# 1. flat-α is an exact fixed point in Cartesian geometry
r0 = BC.run_bdnk_causal(Nr=200, rmax=20.0, τ0=1.0, τfinal=2.0, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=(r->0.0), geom=:cartesian, save_dt=1.0)
fp = maximum(maximum(abs.(a)) for a in r0.αs)
fp < 1e-12 || fail("flat-α not a fixed point: max|α|=$fp")

# 2. Milne charge conservation to machine precision
αp(r) = 0.15*exp(-((r-10.0)/1.0)^2)
rM = BC.run_bdnk_causal(Nr=300, rmax=20.0, τ0=1.0, τfinal=2.5, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=αp, geom=:milne, save_dt=1.0)
Q(qv) = sum(qv)*rM.grid.dr
drift = abs(Q(rM.qs[end]) - Q(rM.qs[1]))/abs(Q(rM.qs[1]))
drift < 1e-10 || fail("charge not conserved: ΔQ/Q=$drift")

# 3. causal support: a pulse stays inside |x| ≤ v_sig·Δt (no superluminal leakage)
αpulse(r) = 0.08*exp(-((r-15.0)/0.6)^2)
Δt = 3.0
rc = BC.run_bdnk_causal(Nr=800, rmax=30.0, τ0=1.0, τfinal=1.0+Δt, DsT=0.24, mode=:kappa,
                        bg=bg, α_init=αpulse, geom=:cartesian, eps_factor=1.0, save_dt=100.0)
g = rc.grid; α = rc.αs[end]; dist = abs.(g.r .- 15.0)
cone = 1.0*Δt + 2.0      # v_sig=c for ε_ν=κ, + pulse half-extent margin
leak = maximum(α[dist .> cone] .|> abs)/maximum(abs.(α))
leak < 1e-5 || fail("acausal leakage outside cone: leak=$leak")
any(isnan, α) && fail("NaN in solution")

println(allpass ? "test_bdnk_causal: PASS (fixed-point, conservation, causality)" :
                  "test_bdnk_causal: FAIL")
exit(allpass ? 0 : 1)
