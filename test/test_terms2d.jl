# ==============================================================================
# test/test_terms2d.jl — gate Gt: the per-term switches of the 2+1D charm sector,
# and the vorticity coupling of the second moment.
#
#   julia --project=Julia/FiVoHydro.jl Julia/FiVoHydro.jl/test/test_terms2d.jl
#
# A switch that exists but does nothing is worse than no switch: an attribution
# study then reports "this term does not matter" about a term it never removed.
# So every switch is tested for being WIRED (turning it off moves the answer), for
# being CLEAN (it removes exactly one term — the per-term differences add up to
# the whole), and for being REFUSED where it cannot act.
#
# Gt1  bookkeeping: `TERMS_2D` lists every `Terms2D` field exactly once, in order,
#      and the defaults are the shipped equations (all on, vorticity off).
# Gt2  first moment: each `fm_*` / `nu_gradalpha` switch removes a nonzero piece, and
#      the pieces sum to the whole source (the row is linear in them).
# Gt3  second moment: the same, channel by channel (xx, xy, yy and the trace).
# Gt4  the VORTICITY coupling 2τ_M π_Q^{λ⟨μ}ω_λ^{ν⟩}, which neither 2-D code carried
#      before 2026-09-10:
#        a  it vanishes identically in the axisymmetric (1-D) limit — which is why
#           every 1-D-limit gate was blind to its absence;
#        b  σ-coupling + ω-coupling == the full contraction π^{λ⟨μ}∇⊥_λu^{ν⟩}, built
#           here by brute-force 3×3 index algebra with the metric. This ties the
#           new term's SIGN and normalisation to the σ coupling, which gate Gm1
#           already holds to the 1-D reduction at 1e-15;
#        c  rotational covariance (a rank-2 transverse tensor: X' = R X Rᵀ);
#        d  it is Δ-traceless (the τ-row from orthogonality closes the trace).
# Gt5  refusal: a switch in a disabled sector, and an unknown name, are errors.
# Gt6  on a SOLVE: default `terms` is bit-identical to no `terms` at all, and every
#      first-moment switch moves the evolved current.
# ==============================================================================

using Printf
using Test
using Random
using LinearAlgebra

const _ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(_ROOT, "main2D.jl"))
using .hydro2d
const H2 = hydro2d

# ------------------------------------------------------------------------------
# a generic flowing state
# ------------------------------------------------------------------------------
function rand_state(rng; ux = 0.6*randn(rng), uy = 0.6*randn(rng))
    s = (; ux, uy, uτ = sqrt(1 + ux^2 + uy^2), τ = 0.6 + 2rand(rng),
         T = 0.18 + 0.3rand(rng),
         dxT = 0.05randn(rng), dyT = 0.05randn(rng), dtT = -0.05 - 0.05rand(rng),
         nux = 0.02randn(rng), nuy = 0.02randn(rng),
         pxx = 0.01randn(rng), pxy = 0.01randn(rng), pyy = 0.01randn(rng),
         PiQ = 0.01randn(rng),
         dta = 0.3randn(rng), dxa = 0.2randn(rng), dya = 0.2randn(rng),
         dtνx = 0.01randn(rng), dtνy = 0.01randn(rng),
         dxνx = 0.01randn(rng), dxνy = 0.01randn(rng), dyνx = 0.01randn(rng), dyνy = 0.01randn(rng),
         dtux = 0.1randn(rng), dtuy = 0.1randn(rng),
         dxux = 0.1randn(rng), dxuy = 0.1randn(rng), dyux = 0.1randn(rng), dyuy = 0.1randn(rng))
    return s
end

"Kinematics derived from a state exactly as `kinematics_2d` builds them."
function kin(s)
    θ  = (s.ux*s.dtux + s.uy*s.dtuy)/s.uτ + s.dxux + s.dyuy + s.uτ/s.τ
    ax = s.uτ*s.dtux + s.ux*s.dxux + s.uy*s.dyux
    ay = s.uτ*s.dtuy + s.ux*s.dxuy + s.uy*s.dyuy
    return θ, ax, ay
end

const EOS = H2.LatticeHRGEOS()

function fm_source(s; terms = H2.Terms2D())
    θ, ax, ay = kin(s)
    _, n, _ = H2.eos_Pne(s.T, -3.0*s.T, EOS)
    dn_dT = H2.hq_dn_dT_2d(s.T, n, EOS)
    h, hp = H2.hq_h_hprime_2d(s.T, EOS)
    Ds = 0.1163/s.T/H2.fmGeV; τn = Ds*h/s.T
    return H2.consistent_fm_source_2d(s.ux, s.uy, s.uτ, s.T, s.dxT, s.dyT, s.dtT,
                                      s.nux, s.nuy, n, dn_dT, τn, Ds, h, hp,
                                      θ, ax, ay, s.dxux, s.dxuy, s.dyux, s.dyuy; terms = terms)
end

function m2_source(s; terms = H2.Terms2D())
    θ, ax, ay = kin(s)
    _, n, _ = H2.eos_Pne(s.T, -3.0*s.T, EOS)
    h, hp = H2.hq_h_hprime_2d(s.T, EOS)
    Ds = 0.1163/s.T/H2.fmGeV; τn = Ds*h/s.T
    τM = H2.tauM_charm_2d(s.T, τn); ηM = H2.ηM_charm_2d(s.T, τn)
    peta, _ = H2.project_shear_traceless_2d(s.ux, s.uy, s.uτ, s.pxx, s.pxy, s.pyy, 0.0)
    nut = H2.nu_tau_2d(s.ux, s.uy, s.uτ, s.nux, s.nuy)
    θν = (s.ux*s.dtνx + s.uy*s.dtνy + s.nux*s.dtux + s.nuy*s.dtuy)/s.uτ -
         nut*(s.ux*s.dtux + s.uy*s.dtuy)/s.uτ^2 + s.dxνx + s.dyνy + nut/s.τ
    aνx = s.uτ*s.dtνx + s.ux*s.dxνx + s.uy*s.dyνx
    aνy = s.uτ*s.dtνy + s.ux*s.dxνy + s.uy*s.dyνy
    r = H2.consistent_m2_source_2d(s.ux, s.uy, s.uτ, s.τ, s.T, s.dxT, s.dyT, s.dtT,
            s.nux, s.nuy, s.pxx, s.pxy, s.pyy, peta, s.PiQ,
            s.dta, s.dxa, s.dya, θν, aνx, aνy, s.dtνx, s.dtνy,
            s.dxνx, s.dxνy, s.dyνx, s.dyνy,
            θ, ax, ay, s.dtux, s.dtuy, s.dxux, s.dxuy, s.dyux, s.dyuy,
            n, τn, Ds, h, hp, τM, ηM, H2.hq_mass(EOS); terms = terms)
    return (r[1], r[2], r[3], r[4]), τM
end

with_off(name) = H2.Terms2D(; NamedTuple{(name,)}((false,))...)
const ALL_OFF_FM = H2.Terms2D(; nu_gradalpha = false, fm_gradT = false, fm_inertial = false,
                                fm_nu_gradu = false, fm_expansion = false, fm_dlnh = false)

# ------------------------------------------------------------------------------
function gate_Gt1()
    println("\nGt1 — the term register is complete and the defaults are the shipped equations")
    names = [t[1] for t in H2.TERMS_2D]
    @test names == collect(fieldnames(H2.Terms2D))
    d = H2.Terms2D()
    for f in fieldnames(H2.Terms2D)
        @test getfield(d, f) == (f !== :m2_vorticity)
    end
    @printf("  %d terms registered, all on by default except m2_vorticity\n", length(names))
    @test :m2_projector in names
    return 0.0
end

# ------------------------------------------------------------------------------
function gate_Gt2()
    println("\nGt2 — first-moment switches are wired and remove exactly one term each")
    rng = MersenneTwister(11)
    fm_terms = (:fm_gradT, :fm_inertial, :fm_nu_gradu, :fm_expansion, :fm_dlnh)
    worst = 0.0
    for _ in 1:40
        s = rand_state(rng)
        full = collect(fm_source(s))
        none = collect(fm_source(s; terms = ALL_OFF_FM))
        @test none == [0.0, 0.0]         # `nu_gradalpha` is not in this function; the rest are
        pieces = [full .- collect(fm_source(s; terms = with_off(k))) for k in fm_terms]
        for p in pieces
            @test maximum(abs, p) > 0     # wired
        end
        rel = maximum(abs, sum(pieces) .- full) / max(maximum(abs, full), 1e-300)
        worst = max(worst, rel)
    end
    @printf("  Σ(per-term pieces) vs whole source: worst rel %.2e over 40 states\n", worst)
    @test worst < 1e-12
    return worst
end

# ------------------------------------------------------------------------------
function gate_Gt3()
    println("\nGt3 — second-moment switches are wired and additive, channel by channel")
    rng = MersenneTwister(12)
    m2_terms = [t[1] for t in H2.TERMS_2D if t[2] === :consistent_m2]
    all_on  = H2.Terms2D(; m2_vorticity = true)
    none = H2.Terms2D(; (k => false for k in m2_terms)...)
    worst = 0.0
    touched = Dict(k => 0.0 for k in m2_terms)
    for _ in 1:40
        s = rand_state(rng)
        full, τM = m2_source(s; terms = all_on)
        base, _ = m2_source(s; terms = none)
        total = collect(full) .- collect(base)
        acc = zeros(4)
        for k in m2_terms
            t = H2.Terms2D(; m2_vorticity = true, NamedTuple{(k,)}((false,))...)
            p = collect(full) .- collect(first(m2_source(s; terms = t)))
            touched[k] = max(touched[k], maximum(abs, p))
            acc .+= p
        end
        scale = max(maximum(abs, collect(full)), 1e-300)
        worst = max(worst, maximum(abs, acc .- total)/scale)
        # with every term off, only the relaxation −X/(τ_M u^τ) is left
        @test isapprox(base[1], -s.pxx/(τM*s.uτ); rtol = 1e-12)
        @test isapprox(base[4], -s.PiQ/(τM*s.uτ); rtol = 1e-12)
    end
    for k in m2_terms
        @printf("  %-15s  max |piece| = %.3e\n", k, touched[k])
        @test touched[k] > 0
    end
    @printf("  Σ(per-term pieces) vs whole: worst rel %.2e over 40 states\n", worst)
    @test worst < 1e-11
    return worst
end

# ------------------------------------------------------------------------------
# Gt4 — the vorticity coupling
# ------------------------------------------------------------------------------
"The ω piece of the LHS source, as `S(ω on) − S(ω off)` in the (xx, xy, yy) channels."
function omega_piece(s)
    on,  τM = m2_source(s; terms = H2.Terms2D(; m2_vorticity = true))
    off, _  = m2_source(s)
    den = τM*s.uτ
    # the function returns −S/(τ_M u^τ), so S_on − S_off = −(r_on − r_off)·τ_M u^τ
    return [-(on[k] - off[k])*den for k in 1:3], τM
end

"Brute-force 3×3 index algebra on (τ, x, y): A^{αν} = ∇⊥^α u^ν, and π_Q^{μν}."
function brute(s)
    g = Diagonal([-1.0, 1.0, 1.0])
    u = [s.uτ, s.ux, s.uy]
    G = zeros(3, 3)                       # G[α, ν] = ∂_α u^ν
    G[1, :] = [(s.ux*s.dtux + s.uy*s.dtuy)/s.uτ, s.dtux, s.dtuy]
    G[2, :] = [(s.ux*s.dxux + s.uy*s.dxuy)/s.uτ, s.dxux, s.dxuy]
    G[3, :] = [(s.ux*s.dyux + s.uy*s.dyuy)/s.uτ, s.dyux, s.dyuy]
    a = G' * u                            # a^ν = u^α ∂_α u^ν
    A = g*G + u*a'                        # ∇⊥^α u^ν  (the derivative index first)
    θ = tr(G) + s.uτ/s.τ
    peta, _ = H2.project_shear_traceless_2d(s.ux, s.uy, s.uτ, s.pxx, s.pxy, s.pyy, 0.0)
    Π = H2.shear_tensor_contravariant_2d(s.ux, s.uy, s.uτ, s.τ, s.pxx, s.pxy, s.pyy, peta)
    P = [Π.tt Π.tx Π.ty; Π.tx Π.xx Π.xy; Π.ty Π.xy Π.yy]
    Δ = inv(g) + u*u'
    σ = (A + A')/2 - Δ*θ/3
    σeta = s.uτ/s.τ - θ/3
    πσ = tr(g*P*g*σ) + peta*σeta          # π_{μν}σ^{μν}; the ηη piece is π^η_η σ^η_η
    return (; g, u, A, P, Δ, θ, πσ, peta)
end

"The ω coupling straight from `vorticity_coupling_2d` — no subtraction, full precision."
function omega_direct(s)
    θ, ax, ay = kin(s)
    peta, _ = H2.project_shear_traceless_2d(s.ux, s.uy, s.uτ, s.pxx, s.pxy, s.pyy, 0.0)
    return collect(H2.vorticity_coupling_2d(s.ux, s.uy, s.uτ, s.τ, s.pxx, s.pxy, s.pyy, peta,
                                            ax, ay, s.dtux, s.dtuy, s.dxux, s.dxuy, s.dyux, s.dyuy))
end

function gate_Gt4()
    println("\nGt4 — the vorticity coupling 2τ_M π_Q^{λ⟨μ}ω_λ^{ν⟩}")
    rng = MersenneTwister(13)

    # a — the 1-D limit: on the +x axis of a radial flow, u^y = 0 and ∂_x u^y = ∂_y u^x = 0
    a_worst = 0.0
    for _ in 1:20
        s = rand_state(rng; uy = 0.0)
        s = merge(s, (; dtuy = 0.0, dxuy = 0.0, dyux = 0.0, nuy = 0.0, pxy = 0.0))
        p, _ = omega_piece(s)
        a_worst = max(a_worst, maximum(abs, p))
    end
    @printf("  a  axisymmetric limit: max |ω piece| = %.2e  (must be exactly zero)\n", a_worst)
    @test a_worst == 0.0

    # b — σ piece + ω piece == the full contraction, by brute force
    b_worst = 0.0
    for _ in 1:40
        s = rand_state(rng)
        pω, τM = omega_piece(s)
        full, _ = m2_source(s)
        offσ, _ = m2_source(s; terms = H2.Terms2D(; m2_pi_sigma = false))
        # the σ piece carries the trace channel too; keep the traceless block
        pσ = [-(full[k] - offσ[k])*τM*s.uτ for k in 1:3]
        B = brute(s)
        X = B.P*B.g*B.A
        X = X + X' - (2B.θ/3)*B.P                     # 2π^{λ⟨μ}∇⊥_λu^{ν⟩} before the Δ-trace
        X = X - (2/3)*B.Δ*B.πσ                        # ... and its Δ-trace removed
        ref = τM .* [X[2,2], X[2,3], X[3,3]]
        @test isapprox(pω, τM .* omega_direct(s); rtol = 1e-9, atol = 1e-14)
        rel = maximum(abs, (pσ .+ pω) .- ref)/max(maximum(abs, ref), 1e-300)
        b_worst = max(b_worst, rel)
    end
    @printf("  b  σ-coupling + ω-coupling vs brute-force π^{λ⟨μ}∇⊥_λu^{ν⟩}: worst rel %.2e\n", b_worst)
    @test b_worst < 1e-12

    # c — rotational covariance of the ω piece
    c_worst = 0.0
    for _ in 1:20
        s = rand_state(rng); φ = 2π*rand(rng)
        R = [cos(φ) -sin(φ); sin(φ) cos(φ)]
        rv(v) = R*v
        u2 = rv([s.ux, s.uy]); ν2 = rv([s.nux, s.nuy]); gT = rv([s.dxT, s.dyT]); ga = rv([s.dxa, s.dya])
        dtu = rv([s.dtux, s.dtuy]); dtν = rv([s.dtνx, s.dtνy])
        Gu = R*[s.dxux s.dxuy; s.dyux s.dyuy]*R'      # ∂'_i u'^j
        Gν = R*[s.dxνx s.dxνy; s.dyνx s.dyνy]*R'
        Pq = R*[s.pxx s.pxy; s.pxy s.pyy]*R'
        s2 = merge(s, (; ux = u2[1], uy = u2[2], nux = ν2[1], nuy = ν2[2], dxT = gT[1], dyT = gT[2],
                       dxa = ga[1], dya = ga[2], dtux = dtu[1], dtuy = dtu[2], dtνx = dtν[1], dtνy = dtν[2],
                       dxux = Gu[1,1], dxuy = Gu[1,2], dyux = Gu[2,1], dyuy = Gu[2,2],
                       dxνx = Gν[1,1], dxνy = Gν[1,2], dyνx = Gν[2,1], dyνy = Gν[2,2],
                       pxx = Pq[1,1], pxy = Pq[1,2], pyy = Pq[2,2]))
        p1 = omega_direct(s); p2 = omega_direct(s2)
        X1 = [p1[1] p1[2]; p1[2] p1[3]]; X2 = [p2[1] p2[2]; p2[2] p2[3]]
        c_worst = max(c_worst, maximum(abs, R*X1*R' - X2)/max(maximum(abs, X1), 1e-300))
    end
    @printf("  c  rotational covariance: worst rel %.2e\n", c_worst)
    @test c_worst < 1e-12

    # d — the ω piece is Δ-traceless: with its τ-row from orthogonality, −X^{ττ} + X^{xx} + X^{yy} = 0
    d_worst = 0.0
    for _ in 1:20
        s = rand_state(rng)
        p = omega_direct(s)
        Xtt = (s.ux^2*p[1] + 2s.ux*s.uy*p[2] + s.uy^2*p[3])/s.uτ^2
        d_worst = max(d_worst, abs(-Xtt + p[1] + p[3])/max(maximum(abs, p), 1e-300))
    end
    @printf("  d  Δ-trace of the ω piece: worst rel %.2e\n", d_worst)
    @test d_worst < 1e-12
    return max(b_worst, c_worst, d_worst)
end

# ------------------------------------------------------------------------------
function gate_Gt5()
    println("\nGt5 — switches that cannot act are refused")
    base = (; eos = H2.LatticeHRGEOS(), enable_diff = true, kappa_coeff = 0.1163)
    @test_throws ErrorException H2.build_model_2d(; base..., terms = (fm_inertial = false,))
    @test_throws ErrorException H2.build_model_2d(; base..., terms = (m2_vorticity = true,))
    @test_throws ErrorException H2.build_model_2d(; base..., terms = (fm_inertia = false,))   # typo
    @test_throws ErrorException H2.build_model_2d(; eos = H2.LatticeHRGEOS(),
                                                   terms = (nu_gradalpha = false,))
    m = H2.build_model_2d(; base..., consistent_fm = true, terms = (fm_inertial = false,))
    @test m.terms.fm_inertial == false && m.terms.fm_gradT == true
    println("  sector-off switch, unknown name, and a typo are all errors; a NamedTuple is accepted")
    io = IOBuffer(); H2.show_equations(io, m); txt = String(take!(io))
    @test occursin("[ ] −τ_n n a^i", txt) && occursin("[x] −κ ∇^⟨i⟩α", txt)
    println("  show_equations marks the switched-off term")
    return 0.0
end

# ------------------------------------------------------------------------------
function gate_Gt6()
    println("\nGt6 — on a solve: default terms are bit-identical, every first-moment switch acts")
    function run(; kw...)
        g = H2.make_grid2d(24, 24; xmax = 6.0, ymax = 6.0)
        m = H2.build_model_2d(; eos = H2.LatticeHRGEOS(), enable_diff = true, kappa_coeff = 0.1163,
                                consistent_fm = true, kw...)
        U = H2.allocate_state(g, m)
        for ix in 1:g.Nxtot, iy in 1:g.Nytot
            x = g.xC[ix]; y = g.yC[iy]
            T = 0.15 + 0.30*exp(-(x^2/1.3 + y^2/0.8)/6)
            H2.set_cell!(U, H2.lin(g, ix, iy), T, -4.0 + 0.8*exp(-((x+1)^2 + y^2)/4), 0.0, 0.0, 0.4, m)
        end
        H2.finalize_ic!(U, g, m; τ0 = 0.4)
        H2.run_sim_2d!(U, g, m; τ0 = 0.4, τfinal = 0.9)
        return U, m.layout
    end
    U0, L = run()
    U1, _ = run(; terms = H2.Terms2D())
    @test isequal(U0, U1)
    println("  terms = Terms2D() reproduces the default bit for bit: ", isequal(U0, U1))
    nu(U) = vcat(U[L.iNux, :], U[L.iNuy, :])
    ref = nu(U0); sc = maximum(abs, ref)
    for k in (:nu_gradalpha, :fm_gradT, :fm_inertial, :fm_nu_gradu, :fm_expansion, :fm_dlnh)
        Uk, _ = run(; terms = NamedTuple{(k,)}((false,)))
        d = maximum(abs, nu(Uk) .- ref)/sc
        @printf("  %-13s off moves ν by %.3e of max|ν|\n", k, d)
        @test d > 1e-6
    end
    return 0.0
end

function main()
    @testset "Gt: term switches (2+1D)" begin
        gate_Gt1(); gate_Gt2(); gate_Gt3(); gate_Gt4(); gate_Gt5(); gate_Gt6()
    end
end
main()
